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
from aleph_experiment import AlephExperiment

from partial_grounding_rules import run_step_partial_grounding_rules
from partial_grounding_aleph import run_step_partial_grounding_aleph,run_step_partial_grounding_hard_rules
from optimize_smac import run_smac
from instance_set import InstanceSet, select_instances_from_runs
from utils import SaveModel

from downward import suites

# All time limits are in seconds
FAST_TIME_LIMITS = {
    'run_experiment' : 10,
    'train-hard-rules' : 60, # Needs to be divided across all schemas
    'smac-optimization-hard-rules' : 300
}

# All time limits are in seconds
MEDIUM_TIME_LIMITS = {
    'run_experiment' : 60, # One minute
    'train-hard-rules' : 600, # Needs to be divided across all schemas
    'smac-optimization-hard-rules' : 300
}

# All time limits are in seconds
TIME_LIMITS_IPC_SINGLE_CORE = {
    'run_experiment' : 10*60, # 10 minutes
    'train-hard-rules' : 60*60, # 1 hour, needs to be divided across all schemas
    'smac-optimization-hard-rules' : 60*60 # 1 hour
}

# All time limits are in seconds
TIME_LIMITS_IPC_MULTICORE = {
    'run_experiment' : 30*60,
    'train-hard-rules' : 60*60, # 1 hour, needs to be divided across all schemas
    'smac-optimization-hard-rules' : 60*60 # 1 hour
}

TIME_LIMITS_SEC = MEDIUM_TIME_LIMITS

# Memory limits are in MB
MEMORY_LIMITS_MB = {
    'run_experiment' : 1024*4,
    'train-hard-rules' : 1024*4
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", help="path to domain file. Alternatively, just provide a path to the directory with a domain.pddl and instance files.")
    parser.add_argument("problem", nargs="*", help="path to problem(s) file. Empty if a directory is provided.")
    parser.add_argument("--domain_knowledge_file", help="path to store knowledge file.")

    parser.add_argument("--path", default='./data', help="path to store results")
    parser.add_argument("--cpus", type=int, default=1, help="number of cpus available")
    parser.add_argument("--total_time_limit", default=30, type=int, help="time limit")
    parser.add_argument("--total_memory_limit", default=7*1024, help="memory limit")
    parser.add_argument("--resume", action="store_true", help="if true, do not delete intermediate files (not recommended for final runs)")

    return parser.parse_args()

def main():
    args = parse_args()

    ROOT = os.path.dirname(os.path.abspath(__file__))

    TRAINING_DIR=args.path

    REPO_GOOD_OPERATORS = f"{ROOT}/fd-symbolic"
    REPO_LEARNING = f"{ROOT}/learning"
    BENCHMARKS_DIR = f"{TRAINING_DIR}/instances"
    REPO_PARTIAL_GROUNDING = f"{ROOT}/fd-partial-grounding"

    save_model = SaveModel(args.domain_knowledge_file)

    if not args.resume:
        if os.path.exists(TRAINING_DIR):
            shutil.rmtree(TRAINING_DIR)
        os.mkdir(TRAINING_DIR)

    if args.resume and os.path.exists(BENCHMARKS_DIR):
        pass # TODO: Assert that instances are the same as before
    else:
        # Copy all input benchmarks to the directory
        if os.path.isdir(args.domain): # If the first argument is a folder instead of a domain file
            shutil.copytree(args.domain, BENCHMARKS_DIR)
            args.domain += "/domain.pddl"
        else:
            os.mkdir(BENCHMARKS_DIR)
            shutil.copy(args.domain, BENCHMARKS_DIR)
            for problem in args.problem:
                shutil.copy(problem, BENCHMARKS_DIR)

    ENV = LocalEnvironment(processes=args.cpus)
    SUITE_ALL = suites.build_suite(TRAINING_DIR, ['instances'])

    # Overall time limit is 10s and 1G
    RUN = RunExperiment (TIME_LIMITS_SEC ['run_experiment'], MEMORY_LIMITS_MB['run_experiment'])


    ###
    # Run lama and symbolic search to gather all training data
    ###

    if not os.path.exists(f'{TRAINING_DIR}/runs-lama'):
        # Run lama, with empty config and using the alias
        ## TODO: enable h2 preprocessor
        RUN.run_planner(f'{TRAINING_DIR}/runs-lama', REPO_PARTIAL_GROUNDING, [], ENV, SUITE_ALL, driver_options = ["--alias", "lama-first"])
    else:
        assert args.resume
    instances_manager = InstanceSet(f'{TRAINING_DIR}/runs-lama')

    # We run the good operators tool only on instances solved by lama in less than 30 seconds
    instances_to_run_good_operators = instances_manager.select_instances([lambda i, p : p['search_time'] < 30])

    SUITE_GOOD_OPERATORS = suites.build_suite(TRAINING_DIR, [f'instances:{name}.pddl' for name in instances_to_run_good_operators])
    if not os.path.exists(f'{TRAINING_DIR}/good-operators-unit'):
        RUN.run_good_operators(f'{TRAINING_DIR}/good-operators-unit', REPO_GOOD_OPERATORS, ['--search', "sbd(store_operators_in_optimal_plan=true, cost_type=1)"], ENV, SUITE_GOOD_OPERATORS)
    else:
        assert args.resume
    instances_manager.add_training_data(f'{TRAINING_DIR}/good-operators-unit')

    has_action_cost = len(select_instances_from_runs(f'{TRAINING_DIR}/good-operators-unit', lambda p : p['use_metric'])) > 0
    if has_action_cost:
        if not os.path.exists(f'{TRAINING_DIR}/good-operators-cost'):
            RUN.run_good_operators(f'{TRAINING_DIR}/good-operators-cost', REPO_GOOD_OPERATORS, ['--search', "sbd(store_operators_in_optimal_plan=true)"], ENV, SUITE_GOOD_OPERATORS)
        else:
            assert args.resume
        instances_manager.add_training_data(f'{TRAINING_DIR}/good-operators-cost')

    TRAINING_INSTANCES = instances_manager.split_training_instances()

    #####
    ## Training of partial grounding hard rules
    #####
    if not os.path.exists(f'{TRAINING_DIR}/partial-grounding-hard-rules'):

        aleph_experiment = AlephExperiment(REPO_LEARNING, args.domain, time_limit=TIME_LIMITS_SEC ['train-hard-rules'], memory_limit=MEMORY_LIMITS_MB ['train-hard-rules'])
        aleph_experiment.run_aleph_hard_rules (f'{TRAINING_DIR}/partial-grounding-hard-rules', instances_manager.get_training_datasets(), ENV)
    else:
        assert args.resume

    exit()

    ### SMAC Optimization to select good sets of good and hard rules
    ### No incremental grounding
    ### full grounding + bad rules

    # We want to fix completely the hard rules at this stage, so let's use all SMAC_INSTANCES
    SMAC_INSTANCES = instances_manager.get_smac_instances(['translator_operators', 'translator_facts', 'translator_variables'])

    run_smac_hard_rules(f'{TRAINING_DIR}', f'{TRAINING_DIR}/smac-hard-rules', args.domain, BENCHMARKS_DIR, SMAC_INSTANCES,
                        walltime_limit=TIME_LIMITS['smac-optimization-hard-rules'], n_trials=10000, n_workers=cpus)

    ####
    # Training of priority partial grounding models
    ####
    #TODO: set time and memory limits
    #TODO: train also without good operators
    for training_data_set in instances_manager.get_training_datasets():
        run_step_partial_grounding_rules(REPO_LEARNING, training_data_set, f'{TRAINING_DIR}/partial-grounding-sklearn', args.domain)
        run_step_partial_grounding_aleph(REPO_LEARNING, training_data_set, f'{TRAINING_DIR}/partial-grounding-aleph', args.domain)


    run_smac(f'{TRAINING_DIR}', f'{TRAINING_DIR}/smac1', args.domain, BENCHMARKS_DIR, SMAC_INSTANCES_FIRST_OPTIMIZATION, walltime_limit=100, n_trials=100, n_workers=cpus)

    save_model.save(os.path.join(TRAINING_DIR, 'smac1', 'incumbent'))


    ###
    # Gather training data for search pruning rules
    ###

    if not os.path.exists(f'{TRAINING_DIR}/runs-pruning-rules'): # TODO: Use at least hard rules over here!
        RUN.run_good_operators(f'{TRAINING_DIR}/runs-pruning-rules', REPO_GOOD_OPERATORS,
                               ['--search', "astar(optimal_plans_heuristic(store_operators_in_optimal_plan=true,store_relaxed_plan=true, cost_type=1), cost_type=1)"],
                               ENV, SUITE_GOOD_OPERATORS)
    else:
        assert args.resume



    RUN.run_planner(f'{TRAINING_DIR}/runs-incumbent', REPO_PARTIAL_GROUNDING, [], ENV, SUITE_ALL, driver_options = [use_config_from_incumbent])
    # Select instances that are solved by incumbent in XX seconds

    ####
    # Final SMAC Optimization
    ####

    ## Run a new SMAC optimization, that optimizes for search time, and that also selects search (lama or something else)


if __name__ == "__main__":
    main()
