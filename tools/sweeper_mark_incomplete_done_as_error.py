import sys
import sqlite3
from os.path import join
from datetime import datetime

from deept.utils.debug import my_print

# A run that is DONE but has result_id=-1 never actually finished writing
# its result: mark_done() commits status="DONE" *before* creating/writing
# to run_results, so a crash in between (e.g. the old unquoted-identifier
# bug in create_run_results_table/save_result) leaves a run permanently
# marked done with no result attached, and the sweeper will never revisit
# it on its own. Resetting it to ERROR makes it resumable again; since
# training already completed, resuming just recomputes the best-checkpoint
# summary from the restored summary_manager and retries mark_done - no
# retraining needed.


if __name__ == '__main__':

    sweep_folder = sys.argv[1]
    db_file = join(sweep_folder, 'sweep.db')

    my_print(f'Hi! Marking incomplete DONE runs as ERROR in {db_file}')

    con = sqlite3.connect(db_file)
    cur = con.cursor()

    incomplete = cur.execute(
        'SELECT run_id, run_ident FROM runs WHERE status="DONE" AND result_id=-1'
    )
    incomplete = incomplete.fetchall()

    if not incomplete:
        my_print('No incomplete DONE runs found. Nothing to do.')
        con.close()
        sys.exit()

    for run_id, run_ident in incomplete:
        my_print(f' [INFO] Marking run {run_ident} (run_id={run_id}) as ERROR.')

    cur_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute(
        f'UPDATE runs SET status="ERROR", finished_at="{cur_timestamp}" '
        f'WHERE status="DONE" AND result_id=-1'
    )
    con.commit()
    con.close()

    my_print(f'Marked {len(incomplete)} run(s) as ERROR.')
