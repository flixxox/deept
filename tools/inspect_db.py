import sys
import sqlite3
from os.path import join
from datetime import datetime

from deept.utils.config import Config
from deept.utils.debug import my_print


def has_results(cur):
    res = cur.execute(
        'SELECT name FROM sqlite_master WHERE name="run_results"'
    )

    if res.fetchone() is None:
        return False

    return True

def compute_avg_hours(runs, started_at_idx, finished_at_idx):
    elapsed_avg = 0
    for entry in runs:
        started_at = entry[started_at_idx]
        finished_at = entry[finished_at_idx]

        started_at = datetime.fromisoformat(started_at)
        finished_at = datetime.fromisoformat(finished_at)

        elapsed = (finished_at - started_at).total_seconds()
        elapsed /= 3600

        elapsed_avg += elapsed
    elapsed_avg /= len(runs)
    return elapsed_avg

if __name__ == '__main__':

    # Config

    max_rows_to_print = 35
    scores = ['train_acc', 'eval_acc', 'eval_acc_std', 'eval_fr']

    # Script

    sweep_folder = sys.argv[1]
    db_file = join(sweep_folder, 'sweep.db')
    config_file = join(sweep_folder, 'config.yaml')
    sweep_name = '-'.join(sweep_folder.split('-')[:-1])

    config = Config.parse_config_from_path(config_file)
    best_ind = config['best_checkpoint_indicator']
    best_ind = f'eval_{best_ind}'

    my_print(f'Hi! Inspecing {sweep_name}')
    my_print(f'Db file: {db_file}')

    con = sqlite3.connect(db_file)
    cur = con.cursor()

    if not has_results(cur):
        my_print('No results yet.')
        sys.exit()

    # Results

    results = cur.execute(f'SELECT * FROM run_results ORDER BY {best_ind} DESC')
    results = results.fetchall()
    result_names = [description[0] for description in cur.description]

    result_run_id_idx = result_names.index('run_id')
    result_scores_idxs = [result_names.index(score) for score in scores] 

    # Runs

    runs = cur.execute(f'SELECT * FROM runs WHERE status="DONE"')
    runs = runs.fetchall()
    run_names = [description[0] for description in cur.description]

    run_run_id_idx = run_names.index('run_id')
    run_run_ident_idx = run_names.index('run_ident')
    run_started_at_idx = run_names.index('started_at')
    run_finished_at_idx = run_names.index('finished_at')

    run_run_ids = [run[run_run_id_idx] for run in runs]

    avg_time =  compute_avg_hours(runs, run_started_at_idx, run_finished_at_idx)

    my_print(f'~~~ Already sweept {len(runs)} runs with {avg_time:4.2f} hours per run.')
    my_print("""+======================================+
|   _ \                   |  |         |
|     /   -_) (_-<  |  |  |   _| (_-<  |
|  _|_\ \___| ___/ \_,_| _| \__| ___/  |
+======================================+""")

    rows = [
        ['run_ident'] + scores + ['started_at', 'finished_at']
    ]

    for result in results:
        run_id = result[result_run_id_idx]
        scores = [str(result[score_idx]) for score_idx in result_scores_idxs]
        
        run_idx = run_run_ids.index(run_id)
        run = runs[run_run_ids.index(run_id)]
        run_ident = run[run_run_ident_idx]
        started_at = run[run_started_at_idx]
        finished_at = run[run_finished_at_idx]

        rows.append(
            [run_ident] + scores + [started_at, finished_at]
        )

        if len(rows) > max_rows_to_print:
            break

    lengths = [0 for _ in range(len(rows[0]))]
    for row in rows:
        for i, entry in enumerate(row):
            lengths[i] = max(lengths[i], len(entry))
    lengths = [l+1 for l in lengths]

    for entry, length in zip(rows[0], lengths):
        my_print(
            f'{entry.ljust(length)}', end=''
        )
    my_print('')
    my_print(''.join(['-' for _ in range(sum(lengths))]))

    for row in rows[1:]:
        for entry, length in zip(row, lengths):
            my_print(
                f'{entry.ljust(length)}', end=''
            )
        my_print('')


    con.close()
