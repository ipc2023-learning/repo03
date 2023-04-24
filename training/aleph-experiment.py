import shutil

from lab.experiment import Experiment, Run

import os

from lab.steps import get_step, get_steps_text, Step

from dataclasses import dataclass
from typing import List
from pathlib import Path

class AlephExperiment:

    def __init__(self, time_limit, memory_limit):
        self.time_limit = time_limit
        self.memory_limit = memory_limit




    def run_aleph_hard_rules (self, REPO_LEARNING, RUNS_DIRS, WORKING_DIR, domain_file):
        exp = Experiment(path=WORKING_DIR + "-exp", environment=ENV)


        # TODO:  Add negated and equal predicate?
        aleph_configs = {"good_rules" : ['--op-file', 'good_operators'],
                         "bad_rules" : ['--op-file', 'good_operators', '--prediction-type', 'bad-actions'],
                         }

        for RUNS_DIR in RUNS_DIRS:
            for config_name, config in aleph_configs.items():

                my_working_dir = f'{WORKING_DIR}/{os.path.basename(RUNS_DIR)}-{config_name}'

                print(f"Running: generate-training-data-aleph.py on {my_working_dir}")

                Call([sys.executable, f'{REPO_LEARNING}/learning-aleph/generate-training-data-aleph.py', f'{RUNS_DIR}', my_working_dir] + config,
                     "generate-aleph-files", time_limit=time_limit_per_config, memory_limit=memory_limit).wait()

                aleph_scripts += [os.path.join(my_working_dir, script) for script in os.listdir(my_working_dir) if script.startswith('learn-')]


        for yap_script in aleph_scripts:
            run = exp.add_run()
            run.add_resource('aleph.pl', symlink=True)

            run.add_command(
            "run-aleph",
                ['bash', yap_script],
                time_limit=self.time_limit,
                memory_limit=self.memory_limit,
            )

            run.set_property("id", [yap_script])
            run.set_property("time_limit", self.time_timit)
            run.set_property("memory_limit", self.memory_timit)


        exp.add_parser("aleph-parser.py")

        exp.add_step("build", exp.build)
        exp.add_step("start", exp.start_runs)


        ENV.run_steps(exp.steps)

        process_lab_dir.process_lab_dir(path_exp+ "-exp", path_exp)
        shutil.rmtree(path_exp+ "-exp")


    def run_planner (self, path_exp, planner, config, ENV, SUITE, build_options = [], driver_options = []):
        rev = "ipc2023-classical"
        cached_rev = MockCachedRevision(name='planner', repo=planner, local_rev='default', global_rev=None, build_options=build_options)

        self.run(path_exp, cached_rev, planner, config, ENV, SUITE, build_options=build_options, driver_options=driver_options, extra_parsers =[f"{os.path.dirname(__file__)}/parsers/goodops-parser.py"] )


    def run_good_operators(self, path_exp, planner, config, ENV, SUITE, build_options = [], driver_options = []):
        rev = "ipc2023-classical"
        cached_rev = MockCachedRevision(name='good_operators', repo=planner, local_rev='default', global_rev=None, build_options=build_options)

        self.run(path_exp, cached_rev, planner,config, ENV, SUITE, build_options=build_options, driver_options=driver_options, extra_parsers =[f"{os.path.dirname(__file__)}/parsers/goodops-parser.py"] )
