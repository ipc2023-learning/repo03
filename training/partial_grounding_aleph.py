import os
import sys
from lab.calls.call import Call
import shutil


def run_step_partial_grounding_hard_rules(REPO_LEARNING, RUNS_DIRS, WORKING_DIR, domain_file, time_limit=300, memory_limit = 4*1024*1024, min_time_per_call=10):
    #TODO: check time and memory limit (right now it's taken as a limit per step, and not a limit in total

    os.mkdir(WORKING_DIR)    # TODO: Set to 10k instead of 1k

    # TODO:  Add negated and equal predicate?
    aleph_configs = {"good_rules" : ['--op-file', 'good_operators'],
                     "bad_rules" : ['--op-file', 'good_operators', '--learn-bad '] # --prediction-type bad-actions'],
                     }

    cwd = os.getcwd()

    for RUNS_DIR in RUNS_DIRS:
        time_limit_per_config = max(min_time_per_call, int(time_limit/len(aleph_configs)))

        for config_name, config in aleph_configs.items():

            my_working_dir = f'{WORKING_DIR}/{os.path.basename(RUNS_DIR)}-{config_name}'

            print(f"Running: generate-training-data-aleph.py on {my_working_dir}")

            Call([sys.executable, f'{REPO_LEARNING}/learning-aleph/generate-training-data-aleph.py', f'{RUNS_DIR}', my_working_dir] + config,
                 "generate-aleph-files", time_limit=time_limit_per_config, memory_limit=memory_limit).wait()

            os.chdir(my_working_dir)

            scripts = [script for script in os.listdir('.') if script.startswith('learn-')]

            time_limit_per_script = max(min_time_per_call, int(time_limit_per_config/len(scripts)))
            for filename in scripts:
                Call(['bash', filename], "run-aleph", stdout=filename.replace('learn-', '') + '.log', stderr=filename.replace('learn-', '') + '.err',
                     time_limit=time_limit_per_script, memory_limit=memory_limit).wait()

            os.chdir(cwd)

            Call([sys.executable, f'{REPO_LEARNING}/learning-aleph/parse-aleph-theory.py', my_working_dir],
                 "parse-aleph", stdout=os.path.join(WORKING_DIR, config_name + ".rules"), time_limit=time_limit, memory_limit=memory_limit).wait()


def run_step_partial_grounding_aleph(REPO_LEARNING, RUNS_DIR, WORKING_DIR, domain_file, time_limit=300, memory_limit = 4*1024*1024):
    #TODO: check time and memory limit (right now it's taken as a limit per step, and not a limit in total

    os.mkdir(WORKING_DIR)
    aleph_configs = {
        "class_probability" : ['--op-file', 'good_operators', '--prediction-type', 'class-probability'],
    }

    cwd = os.getcwd()

    for config_name, config in aleph_configs.items():

        Call([sys.executable, f'{REPO_LEARNING}/learning-aleph/generate-training-data-aleph.py', f'{RUNS_DIR}', f'{WORKING_DIR}/{config_name}'] + config,
             "generate-aleph-files", time_limit=time_limit, memory_limit=memory_limit).wait()

        os.chdir(os.path.join(WORKING_DIR, config_name))

        for filename in os.listdir('.'):
            if filename.startswith('learn-'):
                Call(['bash', filename], "run-aleph", stdout=filename.replace('learn-', '') + '.log', stderr=filename.replace('learn-', '') + '.err', time_limit=time_limit, memory_limit=memory_limit).wait()

        os.chdir(cwd)

        Call([sys.executable, f'{REPO_LEARNING}/learning-aleph/parse-aleph-theory.py', os.path.join(WORKING_DIR, config_name), '--class-probability'],
             "parse-aleph", stdout=os.path.join(WORKING_DIR, config_name + ".rules"), time_limit=time_limit, memory_limit=memory_limit).wait()
