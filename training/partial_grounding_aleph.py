import os
import sys
from lab.calls.call import Call
import shutil

def run_step_partial_grounding_aleph(REPO_LEARNING, RUNS_DIR, WORKING_DIR, domain_file, time_limit=300, memory_limit = 4*1024*1024):
    #TODO: check time and memory limit (right now it's taken as a limit per step, and not a limit in total

    os.mkdir(f"{WORKING_DIR}")    # TODO: Set to 10k instead of 1k
    Call([sys.executable, f'{REPO_LEARNING}/learning-aleph/./generate-training-data-aleph.py', f'{RUNS_DIR}', f'{WORKING_DIR}/aleph_files'], "generate-aleph-files", time_limit=time_limit, memory_limit=memory_limit).wait()

    shutil.copy(f'{REPO_LEARNING}/aleph/aleph.pl', f'{WORKING_DIR}/aleph_files')

    Call([sys.executable, f'{REPO_LEARNING}/learning-aleph/learn-class-probability.py', f'{WORKING_DIR}/aleph_files'], "generate-aleph-files", time_limit=time_limit, memory_limit=memory_limit).wait()


    # TODO: Check if rules have been correctly generated. Otherwise, re-generate with smaller size?
