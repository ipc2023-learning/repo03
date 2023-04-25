from ConfigSpace import Categorical, Float, Configuration, ConfigurationSpace, InCondition
from smac import AlgorithmConfigurationFacade, Scenario, HyperparameterOptimizationFacade

from smac.initial_design.default_design import DefaultInitialDesign

from lab.calls.call import Call

import sys
import os
import json
import subprocess
import re
import shutil

from pathlib import PosixPath

from candidate_models import CandidateModels

# from functools import partial

# Hardcoded paths that depend on the trraining part. This could be passed by parameter instead
PARTIAL_GROUNDING_RULES_DIR = 'partial-grounding-sklearn'
PARTIAL_GROUNDING_ALEPH_DIR  = 'partial-grounding-aleph'
PARTIAL_GROUNDING_HARD_RULES_DIR = 'partial-grounding-hard-rules'

# Hardcoded paths
INTERMEDIATE_SMAC_MODELS = 'intermediate-smac-models'


class Eval:
    def __init__(self, DATA_DIR, WORKING_DIR, domain_file, instances_dir, candidate_models, trial_walltime_limit):
        self.DATA_DIR = DATA_DIR
        self.MY_DIR = os.path.dirname(os.path.realpath(__file__))
        self.candidate_models=candidate_models
        self.trial_walltime_limit=trial_walltime_limit

        self.SMAC_MODELS_DIR = os.path.abspath(os.path.join(WORKING_DIR, INTERMEDIATE_SMAC_MODELS))
        if os.path.exists(self.SMAC_MODELS_DIR):
            shutil.rmtree(self.SMAC_MODELS_DIR)
        os.mkdir(self.SMAC_MODELS_DIR)
        self.instances_dir = instances_dir
        self.domain_file = domain_file

        self.regex_total_time = re.compile(rb"INFO\s+Planner time:\s(.+)s", re.MULTILINE)
        self.regex_operators = re.compile(rb"Translator operators:\s(.+)", re.MULTILINE)
        self.regex_plan_cost = re.compile(rb"\[t=.*s, .* KB\] Plan cost:\s(.+)\n", re.MULTILINE)
        self.regex_no_solution = re.compile(rb"\[t=.*KB\] Completely explored state space.*no solution.*", re.MULTILINE)


    def target_function (self, config: Configuration, instance: str, seed: int) -> float:
        if self.candidate_models.is_using_model(config):
            config_name = self.candidate_models.get_unique_model_name(config)
            model_path = os.path.join(self.SMAC_MODELS_DIR, config_name)
            if not os.path.exists(model_path):
                self.candidate_models.copy_model_to_folder(config, model_path, symlink=True)
        else:
            config_name = "---"
            model_path = '.'
            if 'ipc23' in config['queue_type']:
                return 10000000

        extra_parameters = ['--h2-preprocessor', '--alias', config['alias'], '--grounding-queue', config['queue_type'], '--incremental-grounding', '--incremental-grounding-increment-percentage', '20', '--termination-condition', config['termination-condition']]


        instance_file = os.path.join(self.instances_dir, instance + ".pddl")
        assert(os.path.exists(instance_file))

        command=[sys.executable, f'{self.MY_DIR}/../plan-partial-grounding.py', model_path, self.domain_file, instance_file] + extra_parameters
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            output, error_output = proc.communicate(timeout=self.trial_walltime_limit)

            total_time = self.regex_total_time.search(output)
            num_operators = self.regex_operators.search(output)
            plan_cost = self.regex_plan_cost.search(output)

            if total_time and num_operators and plan_cost:
                total_time = float(total_time.group(1))
                num_operators = int(num_operators.group(1))
                plan_cost = int(plan_cost.group(1))
                print (f"Ran {instance} with queue {config['queue_type']} and model {config_name}: time {total_time}, operators {num_operators}, cost {plan_cost}")
                return num_operators
            elif self.regex_no_solution.search(output):
                print (f"Ran {instance} with queue {config['queue_type']} and model {config_name}: not solved due to partial grounding")
                #print(output.decode())
                return 10000000
            else:
                print (f"WARNING: Ran {instance} with queue {config['queue_type']} and model {config_name}: not solved due to unknown reasons")

                print("Output: ", output.decode())
                if error_output:
                    print("Error Output: ", error_output.decode())
                return 10000000
        except subprocess.CalledProcessError:
            print (f"WARNING: Command failed: {' '.join(command)}")
            print (f"Ran {instance} with queue {config['queue_type']} and model {config_name}: not solved due to crash")
            return 10000000

        except subprocess.TimeoutExpired as e:
            proc.kill()
            print (f"Ran {instance} with queue {config['queue_type']} and model {config_name}: not solved due to time limit")
            print(e)
            return 10000000

        except:
            print (f"Error: Command failed: {' '.join(command)}")

            # print("Output: ", output.decode())
            # if error_output:
            #     print("Error Output: ", error_output.decode())
            return 10000000




# Note: default configuration should solve at least 50% of the instances. Pick instances
# with LAMA accordingly. If we run SMAC multiple times, we can use different instances
# set, as well as changing the default configuration each time.
def run_smac_partial_grounding(DATA_DIR, WORKING_DIR, domain_file, instance_dir, instances_with_features : dict, walltime_limit, trial_walltime_limit, n_trials, n_workers):
    DATA_DIR = os.path.abspath(DATA_DIR) # Make sure path is absolute so that symlinks work
    WORKING_DIR = os.path.abspath(WORKING_DIR) # Making path absolute for using SMAC with multiple cores

    ## Configuration Space ##
    ## Define parameters to select models
    os.mkdir(WORKING_DIR)

    ############################
    ### Gather sk_models,  aleph probability_class, good_rule and bad_rule models
    #############################
    candidate_models = CandidateModels()
    candidate_models.load_sk_folder(os.path.join(DATA_DIR,PARTIAL_GROUNDING_RULES_DIR))
    candidate_models.load_aleph_folder(os.path.join(DATA_DIR, PARTIAL_GROUNDING_ALEPH_DIR))
    candidate_models.load_aleph_folder(os.path.join(DATA_DIR, PARTIAL_GROUNDING_HARD_RULES_DIR))


    ############################
    ### Create model parameters
    #############################

    stopping_condition = Categorical(f"termination-condition", ['full', "relaxed", "relaxed5", "relaxed10", "relaxed20"])
    alias = Categorical('alias', ['lama-first'], default='lama-first')
    queue_type = Categorical("queue_type", ["ipc23-single-queue", "ipc23-round-robin", "fifo", "lifo", 'noveltyfifo', 'roundrobinnovelty', 'roundrobin'], default='ipc23-single-queue')
    # TODO if we get proportions of action schemas, we can also add the ipc23-ratio queue;
    # this requires a file "schema_ratios in the --trained-model-folder with line format: stack:0.246087

    parameters = [alias, queue_type, stopping_condition]
    conditions = []

    for schema, models in candidate_models.sk_models_per_action_schema.items():
        m = Categorical(f"model_{schema}", models)
        parameters.append(m)
        conditions.append(InCondition(child=m, parent=queue_type, values=["ipc23-single-queue", "ipc23-round-robin"]))

    for i, r in enumerate(candidate_models.good_rules):
        good = Categorical(f"good_{i}", [False, True], default=True)
        parameters.append(good)
        conditions.append(InCondition(child=good, parent=stopping_condition, values=["relaxed", "relaxed5", "relaxed10", "relaxed20"]))

    for i, r in enumerate(candidate_models.bad_rules):
        parameters.append(Categorical(f"bad{i}", [False, True], default=True))

    cs = ConfigurationSpace(seed=2023) # Fix seed for reproducibility
    cs.add_hyperparameters(parameters)
    cs.add_conditions(conditions)

    evaluator = Eval (DATA_DIR, WORKING_DIR, domain_file, instance_dir, candidate_models, trial_walltime_limit)

    sorted_instances = sorted ([ins for ins in instances_with_features], key=lambda x : instances_with_features[x]['translator_operators'])

    # for a in sorted_instances:
    #     print(a, instances_with_features[a] )

    scenario = Scenario(
        configspace=cs, deterministic=True,
        output_directory=PosixPath(os.path.join(WORKING_DIR, 'smac')),
        walltime_limit=walltime_limit,
        n_trials=n_trials,
        n_workers=n_workers,
        instances=sorted_instances,
        instance_features=instances_with_features
    )

    # Use SMAC to find the best configuration/hyperparameters
    #smac = AlgorithmConfigurationFacade(scenario, evaluator.target_function) ,#initial_design=DefaultInitialDesign(scenario))
    smac = HyperparameterOptimizationFacade(scenario, evaluator.target_function,)

    incumbent_config = smac.optimize()

    print("Chosen configuration: ", incumbent_config)
    if 'ipc23' in incumbent_config['queue_type']:
        candidate_models.copy_model_to_folder(incumbent_config, os.path.join(WORKING_DIR, 'incumbent'), symlink=False )
    else:
        os.mkdir(os.path.join(WORKING_DIR, 'incumbent'))

    with open(os.path.join(WORKING_DIR, 'incumbent', 'config'), 'w') as config_file:
        properties = {k : v for k,v in incumbent_config.get_dictionary().items() if not k.startswith('good') and not k.startswith('bad')}

        properties['bad-rules'] = [i for i, _ in enumerate(candidate_models.bad_rules) if [incumbent_config[f"bad{i}"]]]
        properties['good-rules'] = [i for i, _ in enumerate(candidate_models.good_rules) if [incumbent_config[f"good{i}"]]]

        json.dump(properties, config_file)
