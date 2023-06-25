#! /usr/bin/env python

import argparse
import os
import os.path
import shutil
import sys
import logging
from lab.environments import  LocalEnvironment
import lab.tools
lab.tools.configure_logging()


sys.path.append(f'{os.path.dirname(__file__)}/training')
import training
from run_experiment import RunExperiment
from aleph_experiment import AlephExperiment

from timer import CountdownWCTimer

from partial_grounding_rules import run_step_partial_grounding_rules
from optimize_smac import run_smac_partial_grounding, run_smac_bad_rules, run_smac_search
from instance_set import InstanceSet, select_instances_from_runs,select_instances_from_runs_with_properties
from utils import SaveModel, filter_training_set, combine_training_sets
from incumbent_set import IncumbentSet

from downward import suites

# All time limits are in seconds
FAST_TIME_LIMITS = {
    'run_experiment' : 10,
    'train-hard-rules' : 60, # time per schema
    'smac-optimization-hard-rules' : 60,
    'smac-partial-grounding-total' : 10,
    'smac-partial-grounding-run' : 2,
    'smac-partial-grounding-run-search' : 6,
    'sklearn-step' : 60,
}

# All time limits are in seconds
MEDIUM_TIME_LIMITS = {
    'run_experiment' : 60, # One minute
    'train-hard-rules' : 120, # time per schema
    'smac-optimization-hard-rules' : 300,
    'smac-partial-grounding-total' : 900,
    'smac-partial-grounding-run' : 60,
    'smac-partial-grounding-run-search' : 120,
    'sklearn-step' : 600,
}

# All time limits are in seconds
TIME_LIMITS_IPC_SINGLE_CORE = {
    'run_experiment' : 10*60, # 10 minutes
    'train-hard-rules' : 30*60, # 30 minutes, time per schema
    'smac-optimization-hard-rules' : 60*60, # 1 hour
    'smac-partial-grounding-total' : 60*60, # 1 hour per optimization
    'smac-partial-grounding-run' : 120,
    'smac-partial-grounding-run-search' : 200,
    'sklearn-step' : 900,
}

# All time limits are in seconds
TIME_LIMITS_IPC_MULTICORE = {
    'run_experiment' : 30*60,
    'train-hard-rules' : 60*60, # 1 hour, needs to be divided across all schemas
    'smac-optimization-hard-rules' : 60*60, # 1 hour
    'smac-partial-grounding-total' : 60*60, # 1 hour
    'smac-partial-grounding-run' : 120,
    'smac-partial-grounding-run-search' : 300,
    'sklearn-step' : 900,
}

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

    args = parser.parse_args()

    args.domain = os.path.abspath(args.domain)
    args.problem = [os.path.abspath(p) for p in args.problem]
    if args.domain_knowledge_file:
        args.domain_knowledge_file = os.path.abspath(args.domain_knowledge_file)

    return args


def main():
    args = parse_args()


    ROOT = os.path.dirname(os.path.abspath(__file__))

    TRAINING_DIR=args.path

    REPO_GOOD_OPERATORS = f"{ROOT}/fd-symbolic"
    REPO_LEARNING = f"{ROOT}/learning"
    BENCHMARKS_DIR = f"{TRAINING_DIR}/instances"
    REPO_PARTIAL_GROUNDING = f"{ROOT}/fd-partial-grounding"

    save_model = SaveModel(args.domain_knowledge_file)


    timer = CountdownWCTimer(args.total_time_limit)


    if args.cpus == 1:
        TIME_LIMITS_SEC = TIME_LIMITS_IPC_SINGLE_CORE
    else:
        TIME_LIMITS_SEC = TIME_LIMITS_IPC_MULTICORE

    # TIME_LIMITS_SEC = MEDIUM_TIME_LIMITS #TODO: REMOVE THIS, just for testing!

    if not args.resume:
        if os.path.exists(TRAINING_DIR):
            shutil.rmtree(TRAINING_DIR)
        os.mkdir(TRAINING_DIR)

    if args.resume and os.path.exists(BENCHMARKS_DIR):
        if os.path.isdir(args.domain): # If the first argument is a folder instead of a domain file
            args.domain += "/domain.pddl"
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

    RUN = RunExperiment (TIME_LIMITS_SEC ['run_experiment'], MEMORY_LIMITS_MB['run_experiment'])


    ###
    # Run lama and symbolic search to gather all training data
    ###
    if not os.path.exists(f'{TRAINING_DIR}/runs-lama'):
        logging.info("Running LAMA on all traning instances (remaining time %s)", timer)
        # Run lama, with empty config and using the alias
        RUN.run_planner(f'{TRAINING_DIR}/runs-lama', REPO_PARTIAL_GROUNDING, [], ENV, SUITE_ALL, driver_options = ["--alias", "lama-first",
                                                                                                                   "--transform-task", f"{REPO_PARTIAL_GROUNDING}/builds/release/bin/preprocess-h2",
                                                                                                                   "--transform-task-options", f"h2_time_limit,300"])
    else:
        assert args.resume

    instances_manager = InstanceSet(f'{TRAINING_DIR}/runs-lama')

    # We run the good operators tool only on instances solved by lama in less than 30 seconds
    instances_to_run_good_operators = instances_manager.select_instances([lambda i, p : p['search_time'] < 30])

    SUITE_GOOD_OPERATORS = suites.build_suite(TRAINING_DIR, [f'instances:{name}.pddl' for name in instances_to_run_good_operators])
    if not os.path.exists(f'{TRAINING_DIR}/good-operators-unit'):
        logging.info("Running good operators with unit cost on %d traning instances (remaining time %s)", len(instances_to_run_good_operators), timer)
        RUN.run_good_operators(f'{TRAINING_DIR}/good-operators-unit', REPO_GOOD_OPERATORS, ['--search', "sbd(store_operators_in_optimal_plan=true, cost_type=1)"], ENV, SUITE_GOOD_OPERATORS)
    else:
        assert args.resume
    instances_manager.add_training_data(f'{TRAINING_DIR}/good-operators-unit')

    TRAINING_SET = f'{TRAINING_DIR}/good-operators-unit'

    has_action_cost = len(select_instances_from_runs(f'{TRAINING_DIR}/good-operators-unit', lambda p : p['use_metric']  and p['max_action_cost'] > 1)) > 0
    has_zero_cost_actions = len(select_instances_from_runs(f'{TRAINING_DIR}/good-operators-unit', lambda p : p['min_action_cost'] == 0)) > 0

    # Skip this step if we have less than 18 hours (added due to the time
    #  restriction of 24h instead of 72 for the IPC. If we spent more than 8h
    #  generating the training data, we cannot afford spending more time on
    #  obtaining the training data).
    if has_action_cost and not has_zero_cost_actions and float(timer.remaining_seconds())/float(args.total_time_limit) > 0.66:
        if not os.path.exists(f'{TRAINING_DIR}/good-operators-cost'):
            logging.info("Running good operators with unit cost on %d traning instances (remaining time %s)", len(instances_to_run_good_operators), timer)

            RUN.run_good_operators(f'{TRAINING_DIR}/good-operators-cost', REPO_GOOD_OPERATORS, ['--search', "sbd(store_operators_in_optimal_plan=true)"], ENV, SUITE_GOOD_OPERATORS)
        else:
            assert args.resume

        if not os.path.exists(f'{TRAINING_DIR}/good-operators-combined'):
            combine_training_sets([TRAINING_SET, os.path.join(TRAINING_DIR,'good-operators-cost')], os.path.join(TRAINING_DIR,'good-operators-combined'))
        else:
            assert args.resume

        TRAINING_SET = os.path.join(TRAINING_DIR,'good-operators-combined')
        instances_manager.add_training_data(os.path.join(TRAINING_DIR,'good-operators-combined'))


    TRAINING_INSTANCES = instances_manager.split_training_instances()

    #####
    ## Training of partial grounding hard rules
    #####
    aleph_experiment = AlephExperiment(REPO_LEARNING, args.domain, time_limit=TIME_LIMITS_SEC ['train-hard-rules'], memory_limit=MEMORY_LIMITS_MB ['train-hard-rules'])
    if not os.path.exists(f'{TRAINING_DIR}/partial-grounding-good-rules'):
        logging.info("Learning Aleph good rules (remaining time %s)", timer)

        aleph_experiment.run_aleph_hard_rules (f'{TRAINING_DIR}/partial-grounding-good-rules', TRAINING_SET, ENV, ['good_rules'])
    else:
        assert args.resume

    #####
    ## Remove actions that are matched by good rules from the training data
    #####
    if os.path.exists(f'{TRAINING_DIR}/partial-grounding-good-rules/good_rules.rules'):
        if not os.path.exists(f'{TRAINING_SET}-nogoodrules'):
            filter_training_set(REPO_LEARNING, TRAINING_SET, f'{TRAINING_DIR}/partial-grounding-good-rules/good_rules.rules', f'{TRAINING_SET}-nogoodrules')
        TRAINING_SET = f'{TRAINING_SET}-nogoodrules'


    #####
    ## Learn bad rules
    #####
    if not os.path.exists(f'{TRAINING_DIR}/partial-grounding-bad-rules'):
        logging.info("Learning Aleph bad rules (remaining time %s)", timer)
        aleph_experiment.run_aleph_hard_rules (f'{TRAINING_DIR}/partial-grounding-bad-rules', TRAINING_SET, ENV, ['bad_rules'])
    else:
        assert args.resume

    #####
    ## Filter of bad rules
    #####

    ### SMAC Optimization to select good sets of good and hard rules
    ### No incremental grounding
    ### full grounding + bad rules

    # We want to fix completely the hard rules at this stage, so let's use all SMAC_INSTANCES
    if not os.path.exists(f'{TRAINING_DIR}/partial-grounding-hard-rules'):
        logging.info("Running SMAC to select good and bad rules (remaining time %s)", timer)

        SMAC_INSTANCES = instances_manager.get_smac_instances(['translator_operators', 'translator_facts', 'translator_variables'])

        run_smac_bad_rules(TRAINING_DIR, os.path.join(TRAINING_DIR, 'smac-partial-grounding-bad-rules'), args.domain, BENCHMARKS_DIR, SMAC_INSTANCES, instances_manager.get_instance_properties(),
                           trial_walltime_limit=TIME_LIMITS_SEC['smac-partial-grounding-run'],
                           walltime_limit=TIME_LIMITS_SEC['smac-partial-grounding-total'],
                           n_trials=10000, n_workers=args.cpus)

        incumbent_path = os.path.join(TRAINING_DIR, 'smac-partial-grounding-bad-rules', 'incumbent')

        assert os.path.exists(incumbent_path)
        shutil.copytree(incumbent_path, f'{TRAINING_DIR}/partial-grounding-hard-rules') # Now, this hard rules are set in stone
    else:
        assert args.resume, "Partial grounding hard rules existed, but no --resume option was provided"

    # Currently, our best incumbent is just lama with the bad pruning rules
    incumbent_set = IncumbentSet(TRAINING_DIR, save_model)

    if os.path.exists(os.path.join(TRAINING_DIR, 'smac-partial-grounding-bad-rules', 'incumbent')):
        logging.info("Model with hard rules is completed. Saving as incumbent model (remaining time %s)", timer)

        # This is not entirely accurate, but we avoid running lama with the bad rules
        incumbent_set.add_and_save(os.path.join(TRAINING_DIR, 'smac-partial-grounding-bad-rules', 'incumbent'), select_instances_from_runs_with_properties(f'{TRAINING_DIR}/runs-lama'))

    #####
    ## Remove actions that are matched by bad rules from the training data
    #####
    if os.path.exists(f'{TRAINING_DIR}/partial-grounding-hard-rules/bad_rules.rules'):
        if not os.path.exists(f'{TRAINING_SET}-nobadrules'):
            filter_training_set(REPO_LEARNING, TRAINING_SET, f'{TRAINING_DIR}/partial-grounding-hard-rules/bad_rules.rules', f'{TRAINING_SET}-nobadrules')
        TRAINING_SET = f'{TRAINING_SET}-nobadrules'

    ####
    # Training of priority partial grounding models
    ####
    if not os.path.exists(f'{TRAINING_DIR}/partial-grounding-aleph'):
        logging.info("Learning Aleph probability model (remaining time %s)", timer)
        aleph_experiment.run_aleph_class_probability (f'{TRAINING_DIR}/partial-grounding-aleph', TRAINING_SET, ENV)
    else:
        assert args.resume


    if not os.path.exists(f'{TRAINING_DIR}/partial-grounding-sklearn'):
        logging.info("Learning sklearn models (remaining time %s)", timer)
        run_step_partial_grounding_rules(REPO_LEARNING, instances_manager.get_training_datasets(), f'{TRAINING_DIR}/partial-grounding-sklearn', args.domain, time_limit=TIME_LIMITS_SEC['sklearn-step'])
    else:
        assert args.resume


    ####
    # Final SMAC Optimization
    ####
    index = 0
    attempted_configs = set()
    while True:
        index += 1

        if os.path.exists(f'{TRAINING_DIR}/smac-{index}'):
            assert (args.resume)
        else:
            logging.info("Starting SMAC optimization round %d (remaining time %s)", index, timer)
            os.mkdir(f'{TRAINING_DIR}/smac-{index}')

            if index % 2 == 1:
                logging.info("SMAC optimization minimizing operators (remaining time %s)", timer)

                _, _, best_configs = run_smac_partial_grounding(f'{TRAINING_DIR}', f'{TRAINING_DIR}/smac-{index}/smac-partial-grounding', args.domain, BENCHMARKS_DIR,
                                                                instances_manager.get_instances_smac_partial_grounding(['translator_operators', 'translator_facts', 'translator_variables']),
                                                                instances_manager.get_instance_properties(),
                                                                walltime_limit=TIME_LIMITS_SEC['smac-partial-grounding-total'],
                                                                trial_walltime_limit=TIME_LIMITS_SEC['smac-partial-grounding-run'],
                                                                n_trials=TIME_LIMITS_SEC['smac-partial-grounding-total'], # Limit the number of rounds, as if we did one run per second
                                                                n_workers=args.cpus, seed=2023+index) #TODO use args.cpus

            else:
                best_configs = None # Sometimes try the full optimization

            logging.info("SMAC optimization minimizing running time (remaining time %s)", timer)

            ## Run a new SMAC optimization, that optimizes for search time, and that also selects search (lama or something else)
            # Continue improving the incumbent
            incumbent_dir, incumbent_config, _ = run_smac_search(f'{TRAINING_DIR}', f'{TRAINING_DIR}/smac-{index}/smac-search', args.domain, best_configs, BENCHMARKS_DIR,
                                                                 instances_manager.get_instances_smac_search(['translator_operators', 'translator_facts', 'translator_variables']), instances_manager.get_instance_properties(),
                                                                 walltime_limit=TIME_LIMITS_SEC['smac-partial-grounding-total'],
                                                                 trial_walltime_limit=TIME_LIMITS_SEC['smac-partial-grounding-run-search'],
                                                                 n_trials=TIME_LIMITS_SEC['smac-partial-grounding-total'], # Limit the number of rounds, as if we did one run per second
                                                                 n_workers=args.cpus, seed=2023+index)


            # Trying and/or adding the same config multiple times is silly, so skip this
            if incumbent_config in attempted_configs:
                logging.info("Skipping config chosen by SMAC because it has already been attempted (remaining time %s)", timer)
                continue
            attempted_configs.add(incumbent_config)

            logging.info("Test config chosen by SMAC (remaining time %s)", timer)

            translate_options = ["--translate-options", "--grounding-action-queue-ordering", incumbent_config['queue_type']]

            if "ipc23" in incumbent_config['queue_type']:
                translate_options += ["--batch-evaluation", "--trained-model-folder", incumbent_dir]
                if "ignore-bad-actions" in incumbent_config and incumbent_config["ignore-bad-actions"].lower().strip() == "true":
                    translate_options += ["--ignore-bad-actions"]


            tc = incumbent_config['termination-condition']
            if tc != "full":
                translate_options += ["--termination-condition", "goal-relaxed-reachable"]
                if tc.startswith("relaxed") and len(incumbent_config['termination-condition']) > len("relaxed"):
                    translate_options += ["percentage", tc[len("relaxed"):]]

            # Test current incumbent
            RUN.run_planner(f'{TRAINING_DIR}/smac-{index}/test', REPO_PARTIAL_GROUNDING, translate_options, ENV, SUITE_ALL,
                            driver_options = ["--alias", incumbent_config['alias'],
                                              "--transform-task", f"{REPO_PARTIAL_GROUNDING}/builds/release/bin/preprocess-h2",
                                              "--transform-task-options", f"h2_time_limit,300", "--incremental-grounding",
                                              "--incremental-grounding-search-time-limit", "300", "--incremental-grounding-increment-percentage", "20"])


            # compare to best_incumbents if it is better, then replace solves a problem that was not solved by a configuration before sum of planner times is lower
            incumbent_set.add_and_save(incumbent_dir, select_instances_from_runs_with_properties(f'{TRAINING_DIR}/smac-{index}/test'))


    ###
    # Gather training data for search pruning rules
    ###
    # if not os.path.exists(f'{TRAINING_DIR}/runs-pruning-rules'): # TODO: Use at least hard rules over here!
    #     RUN.run_good_operators(f'{TRAINING_DIR}/runs-pruning-rules', REPO_GOOD_OPERATORS,
    #                            ['--search', "astar(optimal_plans_heuristic(store_operators_in_optimal_plan=true,store_relaxed_plan=true, cost_type=1), cost_type=1)"],
    #                            ENV, SUITE_GOOD_OPERATORS)
    # else:
    #     assert args.resume



if __name__ == "__main__":
    main()
