import sys
import sqlite3
from os.path import join
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

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
                summary[param][value] = [score]
            else:
                summary[param][value].append(score)

    
    for param in summary.keys():
        my_print(f' ~~~ Performance of {param}')

        param_summary = dict(sorted(summary[param].items(), key=lambda item: item[0]))
        for value, scores in param_summary.items():
            score_sum = sum(scores)
            my_print(f'{value} : {len(scores)} {score_sum/len(scores):4.2f}')

    plot_param_score_summary(summary)

def plot_param_score_summary(summary, prefix="param_score_summary"):
    n_params = len(summary)
    fig, axes = plt.subplots(n_params, 1, figsize=(8, 5 * n_params), squeeze=False)
    axes = axes.flatten()

    for idx, (param, value_dict) in enumerate(summary.items()):
        param_values = []
        avg_scores = []
        counts = []
        for value, scores in value_dict.items():
            param_values.append(float(value))
            avg_scores.append(np.mean(scores))
            counts.append(len(scores))

        # Sort by param value for a nice plot
        sorted_items = sorted(zip(param_values, avg_scores, counts))
        param_values, avg_scores, counts = zip(*sorted_items)

        ax1 = axes[idx]
        ax2 = ax1.twinx()

        ax1.plot(param_values, avg_scores, marker='o', linestyle='-', color='royalblue', label='Avg Score')
        ax1.set_ylabel('Average Score', fontsize=12, color='royalblue')
        ax1.tick_params(axis='y', labelcolor='royalblue')

        # Orange line with opacity
        ax2.plot(param_values, counts, marker='s', linestyle='-', color='orange', label='Num Scores', alpha=0.5)
        ax2.set_ylabel('Num Scores', fontsize=12, color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        ax1.set_title(f'Average Score - {param}', fontsize=14)
        ax1.set_xlabel(f'{param}', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=11)

    plt.tight_layout()
    plt.savefig(f"/Users/fschmidt/img/sweep_analysis/{prefix}_all_params.jpg", format='jpg', dpi=150)
    plt.clf()
    plt.cla()
    plt.close()

if __name__ == '__main__':
    sweep_folder = sys.argv[1]
    db_file = join(sweep_folder, 'sweep.db')
    config_file = join(sweep_folder, 'config.yaml')
    sweep_name = '-'.join(sweep_folder.split('-')[:-1])

    config = Config.parse_config_from_path(config_file)
    best_ind = config['best_checkpoint_indicator']

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
