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

def get_result_for_run(run_id, results, result_run_id_index):
    for result in results:
        if result[result_run_id_index] == run_id:
            return result
    raise RuntimeError('Something went wrong!')

def parse_run_ident(run_ident):
    """Takes the run_ident and returns a dict[param_name] = value"""
    ret = {}
    run_ident = run_ident.split('__')
    for part in run_ident:
        name = part.split('_')[:-1]
        name = '_'.join(name)
        value = float(part.split('_')[-1])
        ret[name] = value
    return ret

def parse_run_and_results(runs, run_names, results, result_names):
    """Since runs and results are in different tables, this function
    merges the two into a single table. It can be seen as merging the columns."""
    columns = {}
    columns['started_at'] = []
    columns['finished_at'] = []
    columns['params'] = {}

    exmp_run_ident = runs[0][run_names.index('run_ident')]
    exmp_run_ident = parse_run_ident(exmp_run_ident)
    for param_name, value in exmp_run_ident.items():
            columns['params'][param_name] = []
    
    for result_name in result_names:
        columns[result_name] = []

    results_by_run_id = [result[result_names.index('run_id')] for result in results]

    for run in runs:
        run_id = run[run_names.index('run_id')]
        run_ident = run[run_names.index('run_ident')]
        started_at = run[run_names.index('started_at')]
        finished_at = run[run_names.index('finished_at')]
        
        result = results[results_by_run_id.index(run_id)]
        run_ident = parse_run_ident(run_ident)
        
        for param_name, value in run_ident.items():
            columns['params'][param_name].append(value)
        columns['started_at'].append(started_at)
        columns['finished_at'].append(finished_at)

        for i, score in enumerate(result):
            columns[result_names[i]].append(score)

    return columns

def param_analysis(columns, best_ind):
    summary = {}
    for param in columns['params'].keys():
        column_values = columns['params'][param]

        summary[param] = {}
        for i, value in enumerate(column_values):
            value = str(value)
            score = columns[best_ind][i]
            if value not in summary[param].keys():
                summary[param][value] = [1, score]
            else:
                summary[param][value][0] += 1
                summary[param][value][1] += score

    
    for param in summary.keys():
        my_print(f' ~~~ Performance of {param}')

        param_summary = dict(sorted(summary[param].items(), key=lambda item: item[0]))
        for value, perf in param_summary.items():
            hits = perf[0]
            score_sum = perf[1]
            my_print(f'{value} : {hits} {score_sum/hits:4.2f}')


if __name__ == '__main__':
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

    results = cur.execute(f'SELECT * FROM run_results')
    results = results.fetchall()
    result_names = [description[0] for description in cur.description]

    # Runs

    runs = cur.execute(f'SELECT * FROM runs WHERE status="DONE"')
    runs = runs.fetchall()
    run_names = [description[0] for description in cur.description]

    # Parse

    columns = parse_run_and_results(runs, run_names, results, result_names)

    # Analyze

    param_analysis(columns, best_ind)

    con.close()
