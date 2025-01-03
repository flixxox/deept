import os
import sys
import sqlite3

from deept.utils.debug import my_print

def print_results(res, names):
    lengths = [1 for _ in range(len(names))]
    for i, name in enumerate(names):
        lengths[i] = max(lengths[i], len(str(name)))

    for entry in enumerate(res):
        entry = entry[1]
        for i, column in enumerate(entry):
            lengths[i] = max(lengths[i], len(str(column)))

    lengths = [l+1 for l in lengths]
    total_length = sum(lengths)

    for name, length in zip(names, lengths):
        my_print(
            f'{name.ljust(length)}', end=''
        )
    my_print('')
    my_print(''.join(['-' for _ in range(total_length)]))

    for entry in res:
        for column, length in zip(entry, lengths):
            my_print(
                f'{str(column).ljust(length)}', end=''
            )
        my_print('')

sweep_folder = sys.argv[1]
db_file = os.path.join(sweep_folder, 'sweep.db')
sweep_name = '-'.join(sweep_folder.split('-')[:-1])

my_print(f'Hi! Inspecing {sweep_name}')
my_print(f'Db file: {db_file}')

con = sqlite3.connect(db_file)
cur = con.cursor()

# my_print('~~~~~ Sweeper Info ~~~~~')

# res = cur.execute('SELECT * FROM sweeperinfo')
# res = res.fetchall()
# names = [description[0] for description in cur.description]

my_print('~~~~~ Overview ~~~~~')

res = cur.execute('SELECT run_ident, status, started_at, finished_at, run_id FROM runs')
res = res.fetchall()
names = [description[0] for description in cur.description]

print_results(res, names)

con.close()