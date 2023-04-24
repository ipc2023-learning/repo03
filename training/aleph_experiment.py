import shutil

from lab.experiment import Experiment, Run
from lab.calls.call import Call

import sys
import os

from lab.steps import get_step, get_steps_text, Step

from dataclasses import dataclass
from typing import List
from pathlib import Path

class AlephExperiment:

    def __init__(self, REPO_LEARNING, domain_file, time_limit, memory_limit):
        self.REPO_LEARNING = REPO_LEARNING
        self.domain_file = domain_file
        self.time_limit = time_limit
        self.memory_limit = memory_limit


    def run_aleph_hard_rules (self, WORKING_DIR, RUNS_DIRS, ENV):
        exp = Experiment(path=os.path.join(WORKING_DIR, "exp"), environment=ENV)

        # TODO:  Add negated and equal predicate?
        aleph_configs = {"good_rules" : ['--op-file', 'good_operators', '--prediction-type', 'good-actions'],
                         "bad_rules" : ['--op-file', 'good_operators', '--prediction-type', 'bad-actions'],
                         }

        for RUNS_DIR in RUNS_DIRS:
            for config_name, config in aleph_configs.items():
                my_working_dir = os.path.abspath(f'{WORKING_DIR}/{os.path.basename(RUNS_DIR)}-{config_name}')

                print(f"Running: generate-training-data-aleph.py on {my_working_dir}")

                Call([sys.executable, os.path.join(self.REPO_LEARNING, 'learning-aleph', 'generate-training-data-aleph.py'), f'{RUNS_DIR}', my_working_dir] + config,
                     "generate-aleph-files", time_limit=self.time_limit, memory_limit=self.memory_limit).wait()

                try:
                    aleph_scripts = [script for script in os.listdir(my_working_dir) if script.startswith('learn-')]
                except:
                    print ("Warning: some aleph scripts for learning hard rules failed to be added to the experiment ")
                    aleph_scripts = []
                    pass

                for script in aleph_scripts:
                        run = exp.add_run()

                        run.add_resource('aleph', os.path.join(my_working_dir, 'aleph.pl'), symlink=True)
                        run.add_resource('exec', os.path.join(my_working_dir, script), symlink=True)
                        run.add_resource('bfile', os.path.join(my_working_dir, script[6:] + '.b'), symlink=True)
                        run.add_resource('ffile', os.path.join(my_working_dir, script[6:] + '.f'), symlink=True)
                        run.add_resource('nfile', os.path.join(my_working_dir, script[6:] + '.n'), symlink=True)

                        run.add_command(
                            "run-aleph",
                            ['bash', script],
                            time_limit=self.time_limit,
                            memory_limit=self.memory_limit,
                        )

                        run.set_property("id", [script])
                        run.set_property("time_limit", self.time_limit)
                        run.set_property("memory_limit", self.memory_limit)
                        run.set_property("action_schema", script[6:])
                        run.set_property("action_schema_args", [])
                        run.set_property("config", config_name)

        exp.add_parser(f"{os.path.dirname(__file__)}/parsers/aleph-parser.py")

        exp.add_step("build", exp.build)
        exp.add_step("start", exp.start_runs)

        ENV.run_steps(exp.steps)

        # process_lab_dir.process_lab_dir(path_exp+ "-exp", path_exp)
        # shutil.rmtree(path_exp+ "-exp")
