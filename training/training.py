from collections import namedtuple
import getpass
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import os
import json

from lab import environments, tools
from lab.experiment import Experiment, get_default_data_dir, Run
from lab.environments import SlurmEnvironment
from downward.experiment import FastDownwardExperiment
from lab.steps import get_step, get_steps_text, Step

from downward.reports.absolute import AbsoluteReport

ARGPARSER = tools.get_argument_parser()
ARGPARSER.epilog = "The list of available steps will be added later."
steps_group = ARGPARSER.add_mutually_exclusive_group()
steps_group.add_argument(
    "steps",
    metavar="step",
    nargs="*",
    default=[],
    help="Name or number of a step below. If none is given, print help.",
)

import bz2

from downward import suites

DIR = Path(__file__).resolve().parent
NODE = platform.node()
USER = getpass.getuser()
REMOTE = USER.endswith("@cs.aau.dk")

class TrainingExperiment(Experiment):

    def __init__(self, benchmark_folder, instances = ["instances"], path=None, environment=None):
        Experiment.__init__(self, path=path, environment=environment)
        self.benchmark_folder = benchmark_folder
        self.instances = instances
        self.subexperiment = {}

    def add_substep(self, name, exp, function, *args, **kwargs):
        exp.add_step(name, function, *args, **kwargs)
        self.steps.append(exp.steps[-1])
        if name in self.subexperiment:
            print(f"Error: same name used multiple times: {name}")
            exit()
        self.subexperiment[name] = exp

    # def add_default_steps(self, repo_good_operators, repo_partial_grounding):


        # self.add_step_useful_facts(repo_partial_grounding)


        # for name, multiplier_useless, multiplier_semiuseful in [('both', '10000', '10000'),
        #                                                         ('semi', '10000', '1'),
        #                                                         ('gradual', '10000', '100')]:
        #     self.add_step_good_operators(f"useful-facts-{name}", repo_good_operators, VERSION_GOOD_OPERATORS,
        #                                  ['--preprocess-options', '--cost-useful-facts', 'relaxed_facts', '--multiplier-cost-semiuseful-facts', multiplier_semiuseful, '--multiplier-cost-useless-facts', multiplier_useless, '--search-options', '--search', "sbd(store_operators_in_optimal_plan=true)"],
        #                                  extra_resources = ["relaxed_facts"])




    def add_step_useful_facts(self, repo_partial_grounding, TIME_LIMIT=1800, MEMORY_LIMIT=3000):
        path_exp = f'{self.path}/exp-useful-facts'

        exp = Experiment(path=path_exp)
        exp.add_resource("repo_partial_grounding", repo_partial_grounding, symlink=True)

        exp.add_parser("parsers/usefulfacts-parser.py")

        for task in suites.build_suite(self.benchmark_folder, self.instances):
            run = exp.add_run()

            run.add_resource("domain", task.domain_file, symlink=True)
            run.add_resource("problem", task.problem_file, symlink=True)

            run.add_command(
                "run-useful-facts",
                ["{repo_partial_grounding}/src/powerlifted/powerlifted.py", "-d", "{domain}", "-i", "{problem}", "-s", "gbfs", "-e", "print-useful-facts", "-g", "yannakakis", "--useful-facts-file", "relaxed_facts"],
                time_limit=TIME_LIMIT,
                memory_limit=MEMORY_LIMIT,
            )

            run.set_property("domain", task.domain)
            run.set_property("problem", task.problem)
            run.set_property("algorithm", "powerlifted")

            run.set_property("time_limit", TIME_LIMIT)
            run.set_property("memory_limit", MEMORY_LIMIT)

            run.set_property("id", [task.domain, task.problem])

        self.add_step("build-useful-facts", exp.build)
        self.add_step("start-useful-facts", exp.start_runs)

        self.add_step("fetch-useful-facts", self.fetch_useful_facts, path_exp)

    def fetch_useful_facts(self, lab_dir):
        for direc in os.listdir(lab_dir):
            if direc.startswith("runs") and os.path.isdir("%s/%s" % (lab_dir, direc)):
                for rundir in os.listdir("%s/%s" % (lab_dir, direc)):
                    input_dir = '%s/%s/%s' % (lab_dir, direc, rundir)

                    domain, task = json.load(open('%s/static-properties' % input_dir))['id']
                    task_name = task.replace('.pddl','')

                    output_dir = f'{self.path}/results/{task_name}'

                    self.copy_if_exists(f"{input_dir}/relaxed_facts", output_dir)


    def run_steps(self):
        """Parse the commandline and run selected steps."""
        ARGPARSER.epilog = get_steps_text(self.steps)
        args = ARGPARSER.parse_args()
        if not args.steps:
            ARGPARSER.print_help()
            return
        steps = [get_step(self.steps, name) for name in args.steps]

        if REMOTE:
            subexperiments = set([self.subexperiment[step.name] for step in steps if step.name in self.subexperiment ])
            if len(subexperiments) > 1:
                print ("You can only execute one experiment in the cluster at a time")
                return

            if  any([step.name.startswith("start") for step in steps]):
                env = self.environment
            else:
                env = environments.LocalEnvironment()

            if len(subexperiments) == 1:
                env.exp = subexperiments.pop()
        else:
            env = self.environment

        env.run_steps(steps)


    def compress_file_bz2(self, src, target, level=9):
        if os.path.isfile(src):
            tarbz2contents = bz2.compress(open(src,'rb').read(), level)
            fh = open(target, "wb")
            fh.write(tarbz2contents)
            fh.close()
            os.remove(src)

    def compress_file_xz(self, src, level=9):
        if os.path.isfile(src):
            subprocess.call(["xz", src])


    def copy_if_exists(self, src, target):
        if os.path.isfile(src):
            shutil.copy(src, target)

    def fetch_good_operators(self, lab_dir, name, fetch_everything):
        for direc in os.listdir(lab_dir):
            if direc.startswith("runs") and os.path.isdir("%s/%s" % (lab_dir, direc)):
                for rundir in os.listdir("%s/%s" % (lab_dir, direc)):
                    input_dir = '%s/%s/%s' % (lab_dir, direc, rundir)
                    self.compress_file_bz2 (f"{input_dir}/all_operators", f"{input_dir}/all_operators.bz2")
                    self.compress_file_xz (f"{input_dir}/output")
                    self.compress_file_xz (f"{input_dir}/output.sas")

                    alg, domain, task = json.load(open('%s/static-properties' % input_dir))['id']

                    task_name = task.replace('.pddl','')

                    output_dir = f'{self.path}/results/{task_name}'

                    if fetch_everything:
                        shutil.copytree(input_dir, output_dir)
                    else:
                        self.copy_if_exists(f"{input_dir}/good_operators", f"{output_dir}/good_operators_{name}")
                        self.copy_if_exists(f"{input_dir}/run.log", f"{output_dir}/run-good-operators-{name}.log")
                        self.copy_if_exists(f"{input_dir}/run.err", f"{output_dir}/run-good-operators-{name}.err")
                        self.copy_if_exists(f"{input_dir}/properties", f"{output_dir}/properties-good-operators-{name}")
