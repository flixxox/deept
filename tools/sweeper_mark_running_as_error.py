import sys
import sqlite3
from os.path import join
from datetime import datetime

from deept.utils.debug import my_print


if __name__ == '__main__':

    sweep_folder = sys.argv[1]
    db_file = join(sweep_folder, 'sweep.db')

    my_print(f'Hi! Marking RUNNING runs as ERROR in {db_file}')

    con = sqlite3.connect(db_file)
    cur = con.cursor()

    running = cur.execute('SELECT run_id, run_ident FROM runs WHERE status="RUNNING"')
    running = running.fetchall()

    if not running:
        my_print('No RUNNING runs found. Nothing to do.')
        con.close()
        sys.exit()

    for run_id, run_ident in running:
        my_print(f' [INFO] Marking run {run_ident} (run_id={run_id}) as ERROR.')

    cur_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute(
        f'UPDATE runs SET status="ERROR", finished_at="{cur_timestamp}" WHERE status="RUNNING"'
    )
    con.commit()
    con.close()

    my_print(f'Marked {len(running)} run(s) as ERROR.')
