import shutil

from lab.experiment import Experiment, Run
from lab.calls.call import Call

from collections import defaultdict

import sys
import os

from lab.steps import get_step, get_steps_text, Step

from dataclasses import dataclass
from typing import List
from pathlib import Path

import json


def get_config_parameters(config):
    # TODO:  Add negated and equal predicate?
    CONFIGS = {"good_rules" : ['--op-file', 'good_operators', '--prediction-type', 'good-actions'],
               "bad_rules" : ['--op-file', 'good_operators', '--prediction-type', 'bad-actions'],
               "class_probability" : ['--op-file', 'good_operators', '--prediction-type', 'class-probability'],
               }
    return CONFIGS[config]

class AlephExperiment:
    def __init__(self, REPO_LEARNING, domain_file, time_limit, memory_limit):
        self.REPO_LEARNING = REPO_LEARNING
        self.domain_file = domain_file
        self.time_limit = time_limit
        self.memory_limit = memory_limit

    def run_aleph (self, WORKING_DIR, RUNS_DIR, ENV, aleph_configs):
        exp = Experiment(path=os.path.join(WORKING_DIR, "exp"), environment=ENV)
        num_runs = 0

        for config_name in aleph_configs:
                my_working_dir = os.path.abspath(f'{WORKING_DIR}/{os.path.basename(RUNS_DIR)}-{config_name}')
                print(f"Running: generate-training-data-aleph.py on {my_working_dir}")

                config = get_config_parameters(config_name)

                Call([sys.executable, os.path.join(self.REPO_LEARNING, 'learning-aleph', 'generate-training-data-aleph.py'), f'{RUNS_DIR}', my_working_dir] + config,
                     "generate-aleph-files", time_limit=self.time_limit, memory_limit=self.memory_limit).wait()

                try:
                    aleph_scripts = [script for script in os.listdir(my_working_dir) if script.startswith('learn-')]
                except:
                    print ("Warning: some aleph scripts for learning hard rules failed to be added to the experiment ")
                    aleph_scripts = []
                    pass

                for script in aleph_scripts:
                        num_runs += 1
                        run = exp.add_run()

                        run.add_resource('aleph', os.path.join(my_working_dir, 'aleph.pl'), symlink=True)
                        run.add_resource('exec', os.path.join(my_working_dir, script), symlink=True)
                        run.add_resource('bfile', os.path.join(my_working_dir, script[6:] + '.b'), symlink=True)
                        run.add_resource('ffile', os.path.join(my_working_dir, script[6:] + '.f'), symlink=True)
                        if os.path.exists(os.path.join(my_working_dir, script[6:] + '.n')):
                            run.add_resource('nfile', os.path.join(my_working_dir, script[6:] + '.n'), symlink=True)

                        run.add_command(
                            "run-aleph",
                            ['bash', script],
                            time_limit=self.time_limit,
                            memory_limit=self.memory_limit,
                        )

                        run.set_property("id", [RUNS_DIR, config_name, script])
                        run.set_property("time_limit", self.time_limit)
                        run.set_property("memory_limit", self.memory_limit)
                        run.set_property("action_schema", script[6:])
                        run.set_property("action_schema_args", [])
                        run.set_property("config", config_name)
                        run.set_property("runs_data", RUNS_DIR)

        if num_runs > 0:
            exp.add_parser(f"{os.path.dirname(__file__)}/parsers/aleph-parser.py")

            exp.add_step("build", exp.build)
            exp.add_step("start", exp.start_runs)

            ENV.run_steps(exp.steps)


    def run_aleph_hard_rules (self, WORKING_DIR, RUNS_DIR, ENV, aleph_configs):
        self.run_aleph (WORKING_DIR, RUNS_DIR, ENV, aleph_configs)

        good_rules = set()
        bad_rules = set()
        if os.path.exists(os.path.join(WORKING_DIR, "exp")):
            for direc in os.listdir(os.path.join(WORKING_DIR, "exp")):
                if direc.startswith("runs") and os.path.isdir(os.path.join(WORKING_DIR, "exp", direc)):
                    for rundir in os.listdir(os.path.join(WORKING_DIR, "exp", direc)):
                        # try:
                        input_dir = os.path.join(WORKING_DIR, "exp", direc, rundir)

                        rules = json.load(open('%s/properties' % input_dir))['rules']
                        config = json.load(open('%s/static-properties' % input_dir))['config']
                        if config == "bad_rules":
                            bad_rules.update(rules)
                        else:
                            good_rules.update(rules)
                        # except:
                        #     print ("Warning: Unknown error while gathering rules")


        if good_rules:
            with open(os.path.join(WORKING_DIR,'candidate-good_rules.rules'), 'w') as f:
                f.write("\n".join(list(sorted(good_rules))))

            Call([sys.executable, f'{self.REPO_LEARNING}/learning-sklearn/filter-irrelevant-rules.py', f'{RUNS_DIR}', f'{WORKING_DIR}/candidate-good_rules.rules', f'{WORKING_DIR}/good_rules.rules', '--filter-good-rules'],
                 "filter-good-rules", time_limit=self.time_limit, memory_limit=self.memory_limit,stdout=os.path.join(WORKING_DIR, 'log_filter_candidate_good_rules')).wait()

        if bad_rules:
            with open(os.path.join(WORKING_DIR,'bad_rules.rules'), 'w') as f:
                f.write("\n".join(list(sorted(bad_rules))))



    def run_aleph_class_probability (self, WORKING_DIR, RUNS_DIR, ENV):
        self.run_aleph (WORKING_DIR, RUNS_DIR, ENV, ['class_probability'])

        class_probability_by_dataset = defaultdict(set)
        for direc in os.listdir(os.path.join(WORKING_DIR, "exp")):
            if direc.startswith("runs") and os.path.isdir(os.path.join(WORKING_DIR, "exp", direc)):
                for rundir in os.listdir(os.path.join(WORKING_DIR, "exp", direc)):
                    try:
                        input_dir = os.path.join(WORKING_DIR, "exp", direc, rundir)

                        rule = json.load(open('%s/properties' % input_dir))['class_probability_rule']
                        runs = os.path.basename(json.load(open('%s/static-properties' % input_dir))['runs_data'])
                        class_probability_by_dataset[runs].add(rule)

                    except Exception as e:
                        print (f"Warning: Unknown error while gathering rules {str(e)}")


        for dataset, ruleset in class_probability_by_dataset.items():
            with open(os.path.join(WORKING_DIR,f'class_probability-{dataset}.rules'), 'w') as f:
                f.write("\n".join(list(sorted(ruleset))))
