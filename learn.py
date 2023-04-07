#! /usr/bin/env python

from __future__ import print_function

import argparse
import os
import os.path
import shutil
import sys

from lab.calls.call import Call
from lab.environments import  LocalEnvironment

sys.path.append(f'{os.path.dirname(__file__)}/training')
import training
from good_operator_experiment import run_step_good_operators

from downward import suites

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", help="path to domain file")
    parser.add_argument("problem", nargs="+", help="path to problem file")
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
    BENCHMARKS_DIR = f"{TRAINING_DIR}/instances-training"
    INSTANCES_SMAC = f"{TRAINING_DIR}/instances-smac"
    REPO_PARTIAL_GROUNDING = f"{ROOT}/fd-partial-grounding"

    if os.path.exists(TRAINING_DIR):
        shutil.rmtree(TRAINING_DIR)
    os.mkdir(TRAINING_DIR)

    # Copy all input benchmarks to the directory
    os.mkdir(BENCHMARKS_DIR)
    shutil.copy(args.domain, BENCHMARKS_DIR)

    os.mkdir(INSTANCES_SMAC)
    shutil.copy(args.domain, INSTANCES_SMAC)

    for problem in args.problem:
        # TODO Split instances in some way and only put some on instances smac
        shutil.copy(problem, BENCHMARKS_DIR)
        shutil.copy(problem, INSTANCES_SMAC)

    ENV = LocalEnvironment(processes=args.cpus)
    SUITE_TRAINING = suites.build_suite(TRAINING_DIR, ['instances-training'])

    run_step_good_operators(f'{TRAINING_DIR}/good-operators-unit', REPO_GOOD_OPERATORS, ['--search', "sbd(store_operators_in_optimal_plan=true, cost_type=1)"], ENV, SUITE_TRAINING, fetch_everything=True,)


    #run_step_partial_grounding
    time_limit = 300
    memory_limit = 4*1024*1024

    os.mkdir(f"{TRAINING_DIR}/partial-grounding-rules")    # TODO: Set to 10k instead of 1k
    Call([sys.executable, f'{REPO_LEARNING}/learning-sklearn/generate-exhaustive-feature-rules.py', args.domain, '--runs', f'{TRAINING_DIR}/good-operators-unit', '--rule_size', '7', '--store_rules', f'{TRAINING_DIR}/partial-grounding-rules/rules-exhaustive-1k', '--num_rules','1000'], "generate-rules", time_limit=time_limit, memory_limit=memory_limit).wait()
    # TODO: Check if rules have been correctly generated. Otherwise, re-generate with smaller size?

    Call([sys.executable, f'{REPO_LEARNING}/learning-sklearn/filter-irrelevant-rules.py', '--instances-relevant-rules', '10', f'{TRAINING_DIR}/good-operators-unit', f'{TRAINING_DIR}/partial-grounding-rules/rules-exhaustive-1k', f'{TRAINING_DIR}/partial-grounding-rules/rules-exhaustive-1k-filtered'], "filter-rules", time_limit=time_limit, memory_limit=memory_limit).wait()

    Call([sys.executable, f'{REPO_LEARNING}/learning-sklearn/generate-training-data.py', \
                                         f'{TRAINING_DIR}/good-operators-unit',\
                                         f'{TRAINING_DIR}/partial-grounding-rules/rules-exhaustive-1k-filtered',\
                                         f'{TRAINING_DIR}/partial-grounding-rules/training-data-good-operators-exhaustive-1k-filtered',\
                                         '--op-file', 'good_operators',\
                                         '--max-training-examples', '1000000' # '--num-test-instances TODO Set some test instances
          ], "generate-training-data-1", time_limit=time_limit, memory_limit=memory_limit).wait()



    # TODO: Consider here more feature selection methods, possibly parameterized
    feature_selection_methods = ["DT"]

    for method in feature_selection_methods:
        Call([sys.executable, f'{REPO_LEARNING}/learning-sklearn/feature-selection.py', '--training-folder', f'{TRAINING_DIR}/partial-grounding-rules/training-data-good-operators-exhaustive-1k-filtered', '--selector-type', method], "feature-selection", time_limit=time_limit, memory_limit=memory_limit).wait()

    # Generate training data for all files of useful rules
    useful_rules_files = [f for f in os.listdir( f'{TRAINING_DIR}/partial-grounding-rules/training-data-good-operators-exhaustive-1k-filtered') if f.startswith('useful_rules')]
    for useful_rules_file in useful_rules_files:
        Call([sys.executable, f'{REPO_LEARNING}/learning-sklearn/generate-training-data.py', \
              f'{TRAINING_DIR}/good-operators-unit',\
              f'{TRAINING_DIR}/partial-grounding-rules/training-data-good-operators-exhaustive-1k-filtered/{useful_rules_file}',\
              f'{TRAINING_DIR}/partial-grounding-rules/training-data-good-operators-exhaustive-1k-{useful_rules_file}',\
              '--op-file', 'good_operators',\
              '--max-training-examples', '1000000' # '--num-test-instances TODO Set some test instances
              ], "generate-training-data", time_limit=time_limit, memory_limit=memory_limit).wait()



    training_data_directories = [f for f in os.listdir( f'{TRAINING_DIR}/partial-grounding-rules/') if f.startswith('training-data')]
    learning_methods = [("DT", ["--model-type", "DT"])]
    for training_data_dir in training_data_directories:
        for learning_method_name, learning_method_parameters in learning_methods:

            output_name = f'model_{training_data_dir}.replace("training-data-", "")_{learning_method_name}'
            os.mkdir (f'{TRAINING_DIR}/partial-grounding-rules/{output_name}')
            Call([sys.executable, f'{REPO_LEARNING}/learning-sklearn/train-model.py', \
                  '--training-set-folder', f'{TRAINING_DIR}/partial-grounding-rules/{training_data_dir}',\
                  '--model-folder', f'{TRAINING_DIR}/partial-grounding-rules/{output_name}',
                  ] + learning_method_parameters, "train", time_limit=time_limit, memory_limit=memory_limit).wait()

if __name__ == "__main__":
    main()
