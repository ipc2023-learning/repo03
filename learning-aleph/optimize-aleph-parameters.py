#! /usr/bin/env python3

from ConfigSpace import Constant, Integer, Categorical, Float, Configuration, ConfigurationSpace, InCondition
from smac import HyperparameterOptimizationFacade, Scenario

import smac

from lab.calls.call import Call

import sys
import os
import json
import subprocess
import re
import shutil

from aleph_background import PredictionType
from aleph_yap_file import  write_yap_file

from collections import defaultdict

from itertools import chain, combinations

from pathlib import PosixPath

MY_DIR = os.path.dirname(os.path.realpath(__file__))

def powerset(iterable):
    "powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


class Eval:
    def __init__(self, WORKING_DIR, instance_features, time_limit):
        self.WORKING_DIR = WORKING_DIR
        self.time_limit = time_limit

        self.regex_accuracy = re.compile(r"Accuracy = (.*)", re.MULTILINE)
        self.regex_summary = re.compile(r"\[Training set summary\] \[\[(\d+),(\d+),(\d+),(\d+)\]\]", re.MULTILINE)
        self.regex_rule = re.compile(r'\[Pos cover = (\d+) Neg cover = (\d+)\](.+)')
        self.regex_rule_timeout = re.compile(r'(.+)\[pos cover = (\d+) neg cover = (\d+)\]')

        self.instance_features = instance_features

        self.saved_rules = defaultdict(set)


    def target_function (self, config: Configuration, instance: str, seed: int) -> float:
        directory, executable = instance.split('/')

        # TODO: set time and memory limits:
        memory_limit = 1000

        # cwd = os.getcwd()

        # unique_id = self.smac_history.get_config_id(config)
        unique_id = smac.utils.configspace.get_config_hash(config, chars=10)
        yap_script = os.path.join(self.WORKING_DIR, directory, f'{executable}-tmp-{unique_id}')

        prediction_type = PredictionType.bad_actions if self.instance_features[instance]['prediction-type-bad-actions'] else PredictionType.good_actions
        write_yap_file (yap_script, 'yap', executable.replace('learn-',''), executable.replace('learn-', '') + '.h', prediction_type= prediction_type,
                        extra_parameters=config.get_dictionary())


        run_yap_script(yap_script, self.time_limit)





        # except:
        #     print (f"Error: Aleph call failed")

        #     print("Output: ", output.decode())
        #     if error_output:
        #         print("Error Output: ", error_output.decode())

            # return 10000000



def optimize_aleph_parameters_with_smac (RUNS_DIR, WORKING_DIR, walltime_limit, n_trials, n_workers):
    os.mkdir(WORKING_DIR)

    conditions = []
    parameters = []

    ############################
    ### Parameters to control search
    #############################

    search = Categorical ("search", ['bf', 'df', 'heuristic', 'ibs', 'ils', 'rls', 'scs', 'id', 'ic', 'ar'], default='bf')

    # Parameters for rls
    rls_type = Categorical ("rls_type", ['gsat', 'wsat', 'rrr', 'anneal'])
    tries = Constant ('tries', 1000)
    moves = Integer("moves", bounds=(2, 10))
    conditions += [InCondition(child=param, parent=search, values=["rls"]) for param in [rls_type, tries, moves] ]

    temperature = Float("temperature", bounds=(0.1, 10))
    conditions.append(InCondition(child=temperature, parent=rls_type, values=["anneal"]))
    walk = Float("walk", bounds=(0.01, 0.99))
    conditions.append(InCondition(child=walk, parent=rls_type, values=["wsat"]))


    conditions.append(InCondition(child=rls_type, parent=search, values=["rls"]))
    scs_prob = Float("scs_prob", bounds=(0, 0.99))
    scs_percentile = Integer ("scs_percentile", bounds = (1, 100))
    scs_type = Categorical ("scs_type", ['blind', 'informed'])

    #   scs_sample ommited as we use prob and percentile
    conditions += [InCondition(child=param, parent=search, values=["scs"]) for param in [scs_prob, scs_percentile, scs_type] ]

    evalfn = Categorical ("evalfn", ['coverage', 'compression', 'pbayes', 'accuracy', 'laplace', 'auto_m', 'mestimate', 'entropy', 'gini', 'sd', 'wracc'], default='coverage')

    parameters += [search, rls_type, tries, moves, temperature, walk, scs_prob, scs_percentile, scs_type, evalfn]

    ############################
    ###  parameters to define the search space
    #############################
    i = Integer ("i", bounds = (1, 3), default=2)
    nodes = Constant ('nodes', 5000)
    explore = Categorical("explore", ["true", "false"], default='false')

    parameters += [i, nodes, explore]

    # minacc and minpos are set already by the script
    # minpos = Constant ('minpos', 2)
    # minposfrac not being used

    refine = Categorical ("refine", ['false', 'auto'], default='false')
    lookahead = Integer("lookahead", bounds=(2, 5))
    conditions.append(InCondition(child=lookahead, parent=refine, values=["auto"]))

    parameters += [refine, lookahead]


    ############################
    ###  Generate instances
    #############################

    instances_with_features = {}

    operator_files = ['good_operators']#, 'sas_plan']

    aleph_options = {"prediction-type-bad-actions" : ['--prediction-type', 'bad-actions'],
                     # 'neg' : ['--add-negated-predicates'],
                     # 'equal' : ['--add-equal-predicate']
                     }

    # TODO: set time and memory limits:
    time_limit=10
    memory_limit = 1000

    # For each configuration, we generate a folder, and those are our instances
    for op_file in operator_files:
        for selected_options in powerset(aleph_options.keys()):
            config_name = "-".join([op_file] + list(selected_options))
            config = ['--op-file', op_file] + sum([aleph_options[opt] for opt in selected_options], [])

            print (" ".join(['Running generate-training-data-aleph.py', RUNS_DIR.split('/')[-1], config_name]))

            print(" ".join ([os.path.join(MY_DIR, 'generate-training-data-aleph.py'), RUNS_DIR, os.path.join(WORKING_DIR,config_name)] + config))

            Call([sys.executable, os.path.join(MY_DIR, 'generate-training-data-aleph.py'), RUNS_DIR, os.path.join(WORKING_DIR,config_name)] + config,
                 "generate-aleph-files", time_limit=time_limit, memory_limit=memory_limit).wait()


            for filename in os.listdir(os.path.join(WORKING_DIR,config_name)):
                if filename.startswith('learn-'):
                    instance_name = os.path.join(config_name,filename)
                    instances_with_features[instance_name] = {opt : opt in selected_options for opt in aleph_options}
                    instances_with_features[instance_name]['schema'] = filename



    ############################
    ###  SMAC configuration
    #############################


    cs = ConfigurationSpace(seed=2023) # Fix seed for reproducibility
    cs.add_hyperparameters(parameters)
    cs.add_conditions(conditions)

    evaluator = Eval (WORKING_DIR, instances_with_features, time_limit)

    # print ([ins for ins in instances_with_features])
    # print(instances_with_features)
    scenario = Scenario(
        configspace=cs, deterministic=True,
        output_directory=PosixPath(os.path.join(WORKING_DIR, 'smac')),
        walltime_limit=walltime_limit,
        n_trials=n_trials,
        n_workers=n_workers,
        instances=[ins for ins in instances_with_features],
        instance_features=instances_with_features, min_budget=len(instances_with_features)
    )

    # Use SMAC to find the best configuration/hyperparameters
    smac = HyperparameterOptimizationFacade(scenario, evaluator.target_function)
    # evaluator.smac_history = smac.runhistory
    incumbent = smac.optimize()

    print("Chosen configuration: ", incumbent)


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("runs_folder", help="path to task pddl file")
    argparser.add_argument("store_training_data", help="Directory to store the training data by gen-subdominization-training")
    argparser.add_argument("--op-file", default="sas_plan", help="Filename to determine whether to learn from plans (sas_plan) or the set of optimal_operators")
    argparser.add_argument("--all-ops-file", nargs='+', default="all_operators.bz2", help="path for finding all operators")
    argparser.add_argument("--domain-name", default="domain", help="name of the domain")
    argparser.add_argument("--overall-time-limit", type=int, default="600", help="name of the domain")
    argparser.add_argument("--aleph-directory", help="Directory where aleph can be found")
    argparser.add_argument("--yap", default='yap', help="Command to execute yap")


    options = argparser.parse_args()

    if os.path.exists(options.store_training_data):
        result = input('Output path "{}" already exists. Overwrite (Y/n)?'.format(options.store_training_data))

        if result.lower() not in ['y', 'yes', '']:
            sys.exit()

        shutil.rmtree(options.store_training_data)

    optimize_aleph_parameters_with_smac (options.runs_folder, options.store_training_data, walltime_limit=options.overall_time_limit, n_trials=1000, n_workers=4)
