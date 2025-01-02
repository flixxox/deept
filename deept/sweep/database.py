import yaml
import sqlite3
from os import mkdir
from os.path import isdir, join, isfile
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

        exclude = [
            'experiment_name',
            'config',
            'output_folder',
            'user_code',
            'resume_training',
            'resume_training_from',
            'use_wandb',
            'remove_from_hash'
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

    # Table: Runs

    def create_runs_table(self):
        self.cur.execute(
            f'CREATE TABLE '
            f'runs(run_id INTEGER PRIMARY KEY, run_ident TEXT, started_at TEXT, finished_at TEXT, status TEXT, result_id INT)'
        )

    def is_already_running_or_done(self, run):
        res = self.cur.execute(
            f'SELECT run_id FROM runs '
            f'WHERE run_ident="{run.ident}" AND (status="RUNNING" OR status="DONE")'
        )
        exists = res.fetchone() is not None
        self.update_lastseen()
        return exists

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

    def mark_running(self, run):
        timestamp = self.timestamp()
        self.cur.execute(
            f'INSERT INTO '
            f'runs(run_ident, started_at, finished_at, status, result_id)'
            f'VALUES("{run.ident}", "{timestamp}", "None", "RUNNING", -1)'
        )
        self.con.commit()
        run.run_id = self.cur.lastrowid

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
        keys = self.get_keys_from_run(run)

        sql_string = f'CREATE TABLE run_results(result_id INTEGER PRIMARY KEY, run_id INTEGER'
        for k in keys:
            sql_string += f', {k} REAL'
        sql_string += ')'

        self.cur.execute(sql_string)

    def save_result(self, run):
        keys = self.get_keys_from_run(run)
        values = self.get_values_from_run(run)

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

    def get_keys_from_run(self, run):
        result = run.get_result()
        keys = [f'train_{k}' for k in result['train'].keys()]
        keys += [f'eval_{k}' for k in result['eval'].keys()]
        return keys

    def get_values_from_run(self, run):
        result = run.get_result()
        keys = [v for v in result['train'].values()]
        keys += [v for v in result['eval'].values()]
        return keys


    def disconnect(self):
        self.con.close()