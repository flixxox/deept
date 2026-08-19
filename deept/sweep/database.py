import sys
import yaml
import sqlite3
from os import mkdir
from os.path import isdir, join, isfile, abspath, dirname
from datetime import datetime

from deept.utils.debug import my_print


class SweepDatabase:

    def __init__(self,
        normal_config,
        sweep_folder_root,
        sweep_name,
        hash_config,
        cleanup_after,
        remove_from_hash
    ):
        self.normal_config = normal_config
        self.sweep_folder_root = sweep_folder_root
        self.sweep_name = sweep_name
        self.hash_config = hash_config
        self.remove_from_hash = remove_from_hash
        self.cleanup_after = cleanup_after
        self.already_scheduled_runs = []

        self.con = None
        self.cur = None
        self.completed_runs = 0

    def connect(self):
        if not isdir(self.sweep_folder_root):
            mkdir(self.sweep_folder_root)

        self.set_or_check_codebase_directory()

        exclude = [
            'experiment_name',
            'config',
            'output_folder',
            'user_code',
            'resume_training',
            'resume_training_from',
            'use_wandb',
            'remove_from_hash',
            'number_of_gpus'
            'early_abort',
            'checkpoints_till_abort',
            'early_abort_threshold',
            'checkpoint_strategy',
            'force_resume_of_running_jobs'
        ] + self.remove_from_hash

        sweep_folder = join(self.sweep_folder_root, self.sweep_name)
        if self.hash_config:
            config_hash = self.normal_config.hash(
                exclude=exclude
            )
            sweep_folder = f'{sweep_folder}-{config_hash}'

        if not isdir(sweep_folder):
            self.newly_created = True
            mkdir(sweep_folder)
        else:
            self.newly_created = False
            my_print(f'Sweeper: Connecting to existing db in {sweep_folder}!')

        config_file = join(sweep_folder, 'config.yaml')
        if self.newly_created:
            self.normal_config.dump_to_file(config_file, exclude=exclude)
            my_print(f'Sweeper: Created new sweep db in {sweep_folder}!')
        
        self.sweep_folder = sweep_folder
        self.db_file = join(sweep_folder, 'sweep.db')
        self.con = sqlite3.connect(self.db_file)
        self.cur = self.con.cursor()

        if self.newly_created:
            self.create_sweeperinfo_tables()
            self.create_runs_table()
        self.register_sweeper()
        self.cleanup()
        my_print('Sweeper: Connected!')

    def set_or_check_codebase_directory(self):
        detected_codebase_directory = self.detect_codebase_directory()

        if not self.normal_config.has_key('codebase_directory'):
            self.normal_config['codebase_directory'] = detected_codebase_directory
        else:
            existing_codebase_directory = self.normal_config['codebase_directory']
            if existing_codebase_directory != detected_codebase_directory:
                my_print(
                    f'[Sweeper] This run is executing from codebase directory '
                    f'"{detected_codebase_directory}", but the config already specifies '
                    f'"{existing_codebase_directory}"! All sweepers contributing to the '
                    f'same sweep must run from the same codebase.'
                )

    def detect_codebase_directory(self):
        main_module = sys.modules.get('__main__')
        entry_file = getattr(main_module, '__file__', None)
        if entry_file is None:
            raise RuntimeError('Sweeper: Could not detect codebase directory, "__main__" has no file!')
        return dirname(abspath(entry_file))

    # Table: Runs

    def create_runs_table(self):
        self.cur.execute(
            f'CREATE TABLE '
            f'runs(run_id INTEGER PRIMARY KEY, run_ident TEXT, started_at TEXT, finished_at TEXT, status TEXT, result_id INT, output_folder TEXT)'
        )

    def is_already_running_or_done(self, run):
        res = self.cur.execute(
            f'SELECT run_id FROM runs '
            f'WHERE run_ident="{run.ident}" AND (status="RUNNING" OR status="DONE")'
        )
        exists = res.fetchone() is not None
        self.update_lastseen()
        return exists

    def get_run_status(self, run):
        """Returns the DB's current record for this run's ident (run_id, status,
        output_folder), or None if this run has never been attempted before."""
        res = self.cur.execute(
            f'SELECT run_id, status, output_folder FROM runs '
            f'WHERE run_ident="{run.ident}"'
        )
        row = res.fetchone()
        self.update_lastseen()
        if row is None:
            return None
        return {
            'run_id': row[0],
            'status': row[1],
            'output_folder': row[2],
        }

    def mark_done(self, run):
        # Check that the run has not been cleaned already
        # This can happen if the sweeper has been too slow.
        assert self.is_already_running_or_done(run), 'We have been too slow. Something else is wrong!'

        timestamp = self.timestamp()
        self.cur.execute(
            f'UPDATE runs '
            f'SET status="DONE", finished_at="{timestamp}" WHERE run_id={run.run_id}'
        )
        self.con.commit()

        if not self.run_results_table_exists():
            self.create_run_results_table(run)
        self.save_result(run)

        self.completed_runs += 1
        self.update_run_count()
        self.update_lastseen()
        self.cleanup()

    def mark_running(self, run, output_folder):
        """Marks a run as running. If this run's ident has no prior record, a
        new row is inserted with the given output_folder. If it does have a
        prior record (i.e. it is a run being resumed or restarted from the
        ERROR state, or force-resumed from a stale RUNNING state - see
        Sweeper.force_resume_of_running_jobs), the existing row is updated
        in place instead, so run_id is preserved; output_folder is
        (re)written either way, since a restarted (as opposed to resumed)
        run may use a fresh one."""
        existing = self.get_run_status(run)
        timestamp = self.timestamp()

        if existing is None:
            self.cur.execute(
                f'INSERT INTO '
                f'runs(run_ident, started_at, finished_at, status, result_id, output_folder)'
                f'VALUES("{run.ident}", "{timestamp}", "None", "RUNNING", -1, "{output_folder}")'
            )
            self.con.commit()
            run.run_id = self.cur.lastrowid
        else:
            assert existing['status'] in ('ERROR', 'RUNNING'), (
                f'Sweeper: Tried to mark {run.ident} as running, but it is already '
                f'in status "{existing["status"]}"! This should have been caught earlier.'
            )
            self.cur.execute(
                f'UPDATE runs '
                f'SET status="RUNNING", started_at="{timestamp}", output_folder="{output_folder}" '
                f'WHERE run_id={existing["run_id"]}'
            )
            self.con.commit()
            run.run_id = existing['run_id']

        self.update_lastseen()
        self.cleanup()

    def cleanup(self):
        res = self.cur.execute(
            'SELECT run_id, run_ident, started_at FROM runs WHERE status="RUNNING"'
        )
        res = res.fetchall()

        # Technically it is possible that there is a valid run in the error state.
        # If the questioned sweeper exceeds time, we do the SELECT request from above, and
        # while we are checking, the sweeper finishes, we get a valid run in an error state.
        # But its very unlikely and the run will simply be repeated.

        cur_datetime = datetime.now()
        cur_timestamp = self.timestamp()

        for entry in res:
            run_id = entry[0]
            run_ident = entry[1]
            started_at = entry[2]
            started_at = datetime.fromisoformat(started_at)
            
            passed_hours = (cur_datetime - started_at).total_seconds()
            passed_hours = passed_hours / 3600

            if passed_hours > self.cleanup_after:
                self.cur.execute(
                    f'UPDATE runs '
                    f'SET status="ERROR", finished_at="{cur_timestamp}" WHERE run_id={run_id}'
                )
                my_print(f'Sweeper: Moved run {run_ident} with id {run_id} to the error state since it takes too long!')

    # Table: Run Results

    def run_results_table_exists(self):
        res = self.cur.execute(
            'SELECT name FROM sqlite_master WHERE name="run_results"'
        )
        return res.fetchone() is not None

    def create_run_results_table(self, run):
        keys = run.get_result_keys()

        sql_string = f'CREATE TABLE run_results(result_id INTEGER PRIMARY KEY, run_id INTEGER'
        for k in keys:
            sql_string += f', {k} REAL'
        sql_string += ')'

        self.cur.execute(sql_string)

    def save_result(self, run):
        keys, values = run.get_result_keys_values()

        key_str = 'run_results(run_id'
        for k in keys:
            key_str += f', {k}'
        key_str += ')'

        value_str = f'VALUES({run.run_id}'
        for v in values:
            value_str += f', {round(float(v), 4)}'
        value_str += ')'

        self.cur.execute(
            f'INSERT INTO {key_str} {value_str}'
        )
        self.con.commit()
        result_id = self.cur.lastrowid

        self.cur.execute(
            f'UPDATE runs '
            f'SET result_id={result_id} WHERE run_id={run.run_id}'
        )
        self.con.commit()

    # Table: Sweeper Info

    def create_sweeperinfo_tables(self):
        """The sweeperinfo table stores information about the sweepers"""
        self.cur.execute(
            f'CREATE TABLE '
            f'sweeperinfo(sweeper_id INTEGER PRIMARY KEY, last_seen TEXT, num_runs INTEGER)'
        )

    def register_sweeper(self):
        timestamp = self.timestamp()
        self.cur.execute(
            f'INSERT INTO sweeperinfo(last_seen, num_runs) VALUES("{timestamp}", 0)'
        )
        self.con.commit()
        self.sweeper_id = self.cur.lastrowid
        my_print(f'Sweeper: Registered new sweeper with id {self.sweeper_id}!')

    def update_run_count(self):
        self.cur.execute(
            f'UPDATE sweeperinfo SET num_runs="{self.completed_runs}" WHERE sweeper_id={self.sweeper_id}'
        )
        self.con.commit()

    def update_lastseen(self):
        timestamp = self.timestamp()
        self.cur.execute(
            f'UPDATE sweeperinfo SET last_seen="{timestamp}" WHERE sweeper_id={self.sweeper_id}'
        )
        self.con.commit()

    # General

    def timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def disconnect(self):
        self.con.close()