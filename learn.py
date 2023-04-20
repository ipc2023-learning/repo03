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
from utils import *

from downward import suites

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", help="path to domain file. Alternatively, just provide a path to the directory with a domain.pddl and instance files.")
    parser.add_argument("problem", nargs="*", help="path to problem(s) file. Empty if a directory is provided.")
    parser.add_argument("--domain_knowledge_file", help="path to store knowledge file.")

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

    if os.path.exists(TRAINING_DIR):
        shutil.rmtree(TRAINING_DIR)
    os.mkdir(TRAINING_DIR)

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

    # Overall time limit is 10s and 1G # TODO: Set suitable time and memory limit
    RUN = RunExperiment (10, 1000)

    # Run lama, with empty config and using the alias
    RUN.run_planner(f'{TRAINING_DIR}/runs-lama', REPO_PARTIAL_GROUNDING, [], ENV, SUITE_ALL, driver_options = ["--alias", "lama-first"])

    # We run the good operators tool only on instances solved by lama in less than 30 seconds
    instances = select_instances_from_runs(f'{TRAINING_DIR}/runs-lama', lambda p : p['search_time'] < 30)
    SUITE_GOOD_OPERATORS = suites.build_suite(TRAINING_DIR, [f'instances:{name}.pddl' for name in instances])
    RUN.run_good_operators(f'{TRAINING_DIR}/good-operators-unit', REPO_GOOD_OPERATORS, ['--search', "sbd(store_operators_in_optimal_plan=true, cost_type=1)"], ENV, SUITE_GOOD_OPERATORS)
    RUN_DIRS = [f'{TRAINING_DIR}/good-operators-unit']

    ######
    ### We split instances between training and SMAC optimization following the criteria:
    ###   (1) Any instance not solved by LAMA is for SMAC optimization
    ###   (2) We need to include at least 3 instances that are solved by lama between 10 and 300 seconds
    ###   (3) We want to minimize the overlap with instances for which we have good operators
    ######

    INSTANCES_WITH_TRAINING_DATA = set(select_instances_from_runs(f'{TRAINING_DIR}/good-operators-unit', lambda p : p['coverage']))

    has_action_cost = len(select_instances_from_runs(f'{TRAINING_DIR}/good-operators-unit', lambda p : p['use_metric'])) > 0
    if has_action_cost:
        RUN.run_good_operators(f'{TRAINING_DIR}/good-operators-cost', REPO_GOOD_OPERATORS, ['--search', "sbd(store_operators_in_optimal_plan=true)"], ENV, SUITE_GOOD_OPERATORS)
        RUN_DIRS.append(f'{TRAINING_DIR}/good-operators-cost')
        INSTANCES_WITH_TRAINING_DATA.update(select_instances(f'{TRAINING_DIR}/good-operators-cost', lambda p : p['coverage']))

    if len(INSTANCES_WITH_TRAINING_DATA) < 10:
        print (f"Warning: we do not have good training instances (only {len(INSTANCES_WITH_TRAINING_DATA)} instances were solvable by our optimal planning tool). Consider adding more easy instances to the training set.")

        RUN_DIRS.append(f'{TRAINING_DIR}/runs-lama')
        # TODO: add some instances solved with lama to the INSTANCES_WITH_TRAINING_DATA


    lama_instances = select_instances_from_runs_with_properties(f'{TRAINING_DIR}/runs-lama')

    instances_not_solved_by_lama = select_instances_from_properties(lama_instances, [not_solved()])
    if not instances_not_solved_by_lama:
        print ("Warning: all instances are solved by lama. Make sure your training instances contain difficult instances that are representative of the difficulty of the domain")

    # Here, we have instances reserved for using SMAC (not all of them need to be used). This includes all the instances for which we do not have training data
    SMAC_INSTANCES = set([instance for instance in lama_instances if instance not in INSTANCES_WITH_TRAINING_DATA])


    # Make sure that at least 3 instances are solved by lama under 2 minutes
    if num_instances_from_properties (lama_instances, [planner_time_under(120), in_instanceset(SMAC_INSTANCES) ]) < 3:
        candidate_instances = select_instances_from_properties (lama_instances, [planner_time_under(120), notin_instanceset(SMAC_INSTANCES) ])
        sorted_instances = sorted(candidate_instances, key = lambda x : lama_instances[x]['planner_time'])

        SMAC_INSTANCES.update(sorted_instances[-3:]) # Pick the last three instances

        if len(INSTANCES_WITH_TRAINING_DATA) > 10:
            INSTANCES_WITH_TRAINING_DATA = [ins for ins in INSTANCES_WITH_TRAINING_DATA if ins not in SMAC_INSTANCES]


    while (num_instances_from_properties(lama_instances, [planner_time_under(120), in_instanceset(SMAC_INSTANCES)]) < len(INSTANCES_WITH_TRAINING_DATA)/10):
        candidate_instances = select_instances_from_properties (lama_instances, [planner_time_under(120), notin_instanceset(SMAC_INSTANCES) ])

        if (candidate_instances):
            sorted_instances = sorted(candidate_instances, key = lambda x : lama_instances[x]['planner_time'])

            SMAC_INSTANCES.add(sorted_instances[-1]) # Pick the last instance

            assert len(INSTANCES_WITH_TRAINING_DATA) > 10
            INSTANCES_WITH_TRAINING_DATA = [ins for ins in INSTANCES_WITH_TRAINING_DATA if ins not in SMAC_INSTANCES]

    num_smac_instances_unsolved = num_instances_from_properties(lama_instances, [not_solved(), in_instanceset(SMAC_INSTANCES) ])
    num_smac_instances_under_2m = num_instances_from_properties(lama_instances, [planner_time_under(120), in_instanceset(SMAC_INSTANCES) ])
    num_smac_instances_overlap = num_instances_from_properties(lama_instances, [in_instanceset(INSTANCES_WITH_TRAINING_DATA), in_instanceset(SMAC_INSTANCES) ])

    print (f"After the split, we have {len(INSTANCES_WITH_TRAINING_DATA)} instances for training, {len(SMAC_INSTANCES)} for hyperparameter optimization, out of which {num_smac_instances_overlap} have overlap and {num_smac_instances_unsolved} are not solved by lama and {num_smac_instances_under_2m} are solved under 2 minutes)")
    print ("Instances for the training phase: ", INSTANCES_WITH_TRAINING_DATA)
    print ("Instances for SMAC optimization: ", SMAC_INSTANCES)

    ####
    # run_hard_rules()
    ####

    ### SMAC Optimization to select good sets of good and hard rules
    ### No incremental grounding
    ### full grounding + bad rules



    #TODO: set time and memory limits
    #TODO: train also without good operators
    run_step_partial_grounding_rules(REPO_LEARNING, f'{TRAINING_DIR}/good-operators-unit', f'{TRAINING_DIR}/partial-grounding-rules', args.domain)
    run_step_partial_grounding_aleph(REPO_LEARNING, f'{TRAINING_DIR}/good-operators-unit', f'{TRAINING_DIR}/partial-grounding-aleph', args.domain)


    SMAC_INSTANCES_FIRST_OPTIMIZATION =  select_instances_from_runs_with_properties(f'{TRAINING_DIR}/runs-lama', lambda p : p['problem'] in SMAC_INSTANCES,
                                                                                    ['translator_operators', 'translator_facts', 'translator_variables'])

    # Make sure that we have 10 instances for SMAC. Preferably, instances that were not solved by good operators.
    # Score instances according to how far they are from the desired range, pick 10 best (diversifying)
    assert (len(SMAC_INSTANCES_FIRST_OPTIMIZATION))

    # SMAC_INSTANCES =  select_instances_with_properties(f'{TRAINING_DIR}/runs-lama', lambda p : p['coverage'] == 0, ['translator_operators'])
    # select_instances_by_order (SMAC_INSTANCES, lambda x,y :  x['translator_operators'] < u['translator_operators'] )


    # TODO: check n workers
    run_smac(f'{TRAINING_DIR}', f'{TRAINING_DIR}/smac1', args.domain, BENCHMARKS_DIR, SMAC_INSTANCES_FIRST_OPTIMIZATION, walltime_limit=100, n_trials=100, n_workers=1)

    if args.domain_knowledge_file:
        save_model(os.path.join(TRAINING_DIR, 'smac1', 'incumbent'), args.domain_knowledge_file)


    # RUN.run_planner(f'{TRAINING_DIR}/runs-incumbent', REPO_PARTIAL_GROUNDING, [], ENV, SUITE_ALL, driver_options = [use_config_from_incumbent])
    # Select instances that are solved by incumbent in XX seconds

    ## Run a new SMAC optimization, that optimizes for search time, and that also selects search (lama or something else)


if __name__ == "__main__":
    main()
