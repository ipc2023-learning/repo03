import os
import sys
from lab.calls.call import Call
import shutil



def run_step_partial_grounding_aleph(REPO_LEARNING, RUNS_DIR, WORKING_DIR, domain_file, time_limit=300, memory_limit = 4*1024*1024):
    #TODO: check time and memory limit (right now it's taken as a limit per step, and not a limit in total

    # if os.path.exists(WORKING_DIR):
    #     shutil.rmtree(WORKING_DIR)

    os.mkdir(WORKING_DIR)    # TODO: Set to 10k instead of 1k

    # TODO:  Add negated and equal predicate?
    aleph_configs = {"good_rules" : ['--op-file', 'good_operators'],
                     "bad_rules" : ['--op-file', 'good_operators', '--learn-bad'],
                     "class_probability" : ['--op-file', 'good_operators', '--class-probability'],
    }

    aleph_parse = {"class_probability" : ['--class-probability']}

    cwd = os.getcwd()

    for config_name, config in aleph_configs.items():

        Call([sys.executable, f'{REPO_LEARNING}/learning-aleph/generate-training-data-aleph.py', f'{RUNS_DIR}', f'{WORKING_DIR}/{config_name}'] + config,
             "generate-aleph-files", time_limit=time_limit, memory_limit=memory_limit).wait()

        os.chdir(os.path.join(WORKING_DIR, config_name))

        for filename in os.listdir('.'):
            if filename.startswith('learn-'):
                Call(['bash', filename], "run-aleph", stdout=filename.replace('learn-', '') + '.log', stderr=filename.replace('learn-', '') + '.err', time_limit=time_limit, memory_limit=memory_limit).wait()

        os.chdir(cwd)

        Call([sys.executable, f'{REPO_LEARNING}/learning-aleph/parse-aleph-theory.py', os.path.join(WORKING_DIR, config_name)] + (aleph_parse[config_name] if config_name in aleph_parse else []), "parse-aleph", stdout=os.path.join(WORKING_DIR, config_name + ".rules"), time_limit=time_limit, memory_limit=memory_limit).wait()
