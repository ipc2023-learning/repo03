#! /usr/bin/env python

from __future__ import print_function

import argparse
import os
import os.path
import shutil
import sys

from lab.environments import  LocalEnvironment

sys.path.append(f'{os.path.dirname(__file__)}/training')
import training
from run_experiment import RunExperiment
from partial_grounding_rules import run_step_partial_grounding_rules
from partial_grounding_aleph import run_step_partial_grounding_aleph
from optimize_smac import run_smac
from utils import select_instances_by_properties

from downward import suites

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", help="path to domain file. Alternatively, just provide a path to the directory with a domain.pddl and instance files.")
    parser.add_argument("problem", nargs="*", help="path to problem(s) file. Empty if a directory is provided.")
    parser.add_argument("--path", default='./data', help="path to store results")
    parser.add_argument("--cpus", type=int, default=1, help="number of cpus available")
    parser.add_argument("--total_time_limit", default=30, type=int, help="time limit")
    parser.add_argument("--total_memory_limit", default=7*1024, help="memory limit")

    return parser.parse_args()


def main():
    args = parse_args()

    ROOT = os.path.dirname(os.path.abspath(__file__))

    TRAINING_DIR=args.path

    REPO_GOOD_OPERATORS = f"{ROOT}/fd-symbolic"
    REPO_LEARNING = f"{ROOT}/learning"
    BENCHMARKS_DIR = f"{TRAINING_DIR}/instances"
    REPO_PARTIAL_GROUNDING = f"{ROOT}/fd-partial-grounding"

    # if os.path.exists(TRAINING_DIR):
    #     shutil.rmtree(TRAINING_DIR)
    # os.mkdir(TRAINING_DIR)

    # # Copy all input benchmarks to the directory
    # if os.path.isdir(args.domain): # If the first argument is a folder instead of a domain file
    #     shutil.copytree(args.domain, BENCHMARKS_DIR)
    #     args.domain += "/domain.pddl"
    # else:
    #     os.mkdir(BENCHMARKS_DIR)
    #     shutil.copy(args.domain, BENCHMARKS_DIR)
    #     for problem in args.problem:
    #         shutil.copy(problem, BENCHMARKS_DIR)

    # ENV = LocalEnvironment(processes=args.cpus)
    # SUITE_ALL = suites.build_suite(TRAINING_DIR, ['instances'])

    # # Overall time limit is 10s and 1G # TODO: Set suitable time and memory limit
    # RUN = RunExperiment (10, 1000)

    # # Run lama, with empty config and using the alias
    # RUN.run_planner(f'{TRAINING_DIR}/runs-lama', REPO_PARTIAL_GROUNDING, [], ENV, SUITE_ALL, driver_options = ["--alias", "lama-first"])

    # # We run the good operators tool only on instances solved by lama in less than 30 seconds
    # instances = select_instances_by_properties(f'{TRAINING_DIR}/runs-lama', lambda p : p['search_time'] < 30)
    # SUITE_GOOD_OPERATORS = suites.build_suite(TRAINING_DIR, [f'instances:{name}.pddl' for name in instances])
    # RUN.run_good_operators(f'{TRAINING_DIR}/good-operators-unit', REPO_GOOD_OPERATORS, ['--search', "sbd(store_operators_in_optimal_plan=true, cost_type=1)"], ENV, SUITE_GOOD_OPERATORS)

    # has_action_cost = len(select_instances_by_properties(f'{TRAINING_DIR}/good-operators-unit', lambda p : p['use_metric'])) > 0
    # if has_action_cost:
    #     RUN.run_good_operators(f'{TRAINING_DIR}/good-operators-cost', REPO_GOOD_OPERATORS, ['--search', "sbd(store_operators_in_optimal_plan=true)"], ENV, SUITE_GOOD_OPERATORS)


    # #TODO: set time and memory limits
    # #TODO: train also without good operators
    # run_step_partial_grounding_rules(REPO_LEARNING, f'{TRAINING_DIR}/good-operators-unit', f'{TRAINING_DIR}/partial-grounding-rules', args.domain)

    # run_step_partial_grounding_aleph(REPO_LEARNING, f'{TRAINING_DIR}/good-operators-unit', f'{TRAINING_DIR}/partial-grounding-aleph', args.domain)


    # TODO: Select different instances for instances
    SMAC_INSTANCES =  select_instances_by_properties(f'{TRAINING_DIR}/runs-lama', lambda p : p['search_time'] < 30)
    run_smac(f'{TRAINING_DIR}', args.domain, BENCHMARKS_DIR, SMAC_INSTANCES, walltime_limit=100, n_trials=100, n_workers=1)


if __name__ == "__main__":
    main()
