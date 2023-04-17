from ConfigSpace import Categorical, Float, Configuration, ConfigurationSpace, InCondition

from collections import defaultdict
import numpy as np
from smac import HyperparameterOptimizationFacade, Scenario
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import os
import shutil
from lab.calls.call import Call
import sys

import subprocess
import re

# from functools import partial

# Hardcoded paths that depend on the trraining part. This could be passed by parameter instead
PARTIAL_GROUNDING_RULES_DIR = 'partial-grounding-rules'
PARTIAL_GROUNDING_ALEPH_DIR = 'partial-grounding-aleph'
SUFFIX_ALEPH_MODELS = '.rules'
PREFIX_SK_MODELS = 'model_'

# Hardcoded paths
INTERMEDIATE_SMAC_MODELS = 'intermediate-smac-models'


def copy_model_to_folder(config, sk_models_per_action_schema, DATA_DIR, target_dir, symlink=False ):
    os.mkdir(target_dir)

    collected_relevant_rules = []
    collected_aleph_models = []
    for aschema in sk_models_per_action_schema:
        if config[f'model_{aschema}'].startswith(PREFIX_SK_MODELS):
            assert os.path.exists(os.path.join(DATA_DIR,PARTIAL_GROUNDING_RULES_DIR, config[f'model_{aschema}'], aschema))

            if symlink:
                os.symlink(os.path.join(DATA_DIR,PARTIAL_GROUNDING_RULES_DIR, config[f'model_{aschema}'], aschema), os.path.join(target_dir, aschema))
            else:
                shutil.copy(os.path.join(DATA_DIR,PARTIAL_GROUNDING_RULES_DIR, config[f'model_{aschema}'], aschema), os.path.join(target_dir, aschema))

            with open(os.path.join(DATA_DIR,PARTIAL_GROUNDING_RULES_DIR, config[f'model_{aschema}'], 'relevant_rules')) as rfile:
                for line in rfile:
                    if line.startswith (aschema[:-6] + " ("):
                        collected_relevant_rules.append(line.strip())
        # elif config[f'model_{aschema}'].endswith(SUFFIX_ALEPH_MODELS):
            # with open(os.path.join(DATA_DIR, ALEPH_DIR, model_flename)) as probability_model:
            #     for line in probability_model.readlines():
            #         schema = line.split(":-")[0].strip()
            #         if schema == aschema:
            #             collected_aleph_models.append(model_filename)

        else:
            assert config[f'model_{aschema}'] == 'none'

    if collected_relevant_rules:
        with open(os.path.join(target_dir, 'relevant_rules'), 'w') as f:
            f.write('\n'.join(collected_relevant_rules))

    if collected_aleph_models:
        with open(os.path.join(target_dir, 'probability_class.rules'), 'w') as f:
            f.write('\n'.join(collected_aleph_models))


class Eval:
    def __init__(self, DATA_DIR, WORKING_DIR, domain_file, instances_dir, sk_models_per_action_schema):
        self.DATA_DIR = os.path.abspath(DATA_DIR)
        self.MY_DIR = os.path.dirname(os.path.realpath(__file__))
        self.sk_models_per_action_schema=sk_models_per_action_schema

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

    def get_unique_model_name(self, config):
        assert all([f'model_{aschema}' in config for aschema in self.sk_models_per_action_schema])
        prefix = lambda x : "sk" if x.startswith(PREFIX_SK_MODELS) else ("a" if x.endswith(SUFFIX_ALEPH_MODELS) else "")
        return "-".join([prefix(config[f'model_{aschema}']) + str(opts.index(config[f'model_{aschema}'])) for aschema, opts in self.sk_models_per_action_schema.items()])


    def target_function (self, config: Configuration, instance: str, seed: int) -> float:
        # create folder for the model
        using_model = all ([f'model_{aschema}' in config for aschema in self.sk_models_per_action_schema]) and \
            any  ([config[f'model_{aschema}'] != 'none' for aschema in self.sk_models_per_action_schema])

        if using_model:
            config_name = self.get_unique_model_name(config)
            model_path = os.path.join(self.SMAC_MODELS_DIR, config_name)
            if not os.path.exists(model_path):
                copy_model_to_folder(config, self.sk_models_per_action_schema, self.DATA_DIR, model_path, symlink=True)
        else:
            config_name = "---"
            model_path = '.'
            if 'trained' in config['queue_type']:
                return 10000000


        extra_parameters = ['--h2-preprocessor', '--alias', config['alias'], '--grounding-queue', config['queue_type']]


        instance_file = os.path.join(self.instances_dir, instance + ".pddl")
        assert(os.path.exists(instance_file))

        command=[sys.executable, f'{self.MY_DIR}/../plan-partial-grounding.py', model_path, self.domain_file, instance_file] + extra_parameters
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        try:
            output, error_output = proc.communicate(timeout=300) # Timeout in seconds TODO: set externally

            total_time = self.regex_total_time.search(output)
            num_operators = self.regex_operators.search(output)
            plan_cost = self.regex_plan_cost.search(output)

            if total_time and num_operators and plan_cost:
                total_time = float(total_time.group(1))
                num_operators = float(num_operators.group(1))
                plan_cost = float(plan_cost.group(1))
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

        except subprocess.TimeoutExpired:
            proc.kill()
            print (f"Ran {instance} with queue {config['queue_type']} and model {config_name}: not solved due to time limit")
            return 10000000

        except:
            print (f"Error: Command failed: {' '.join(command)}")

            print("Output: ", output.decode())
            if error_output:
                print("Error Output: ", error_output.decode())







# Note: default configuration should solve at least 50% of the instances. Pick instances
# with LAMA accordingly. If we run SMAC multiple times, we can use different instances
# set, as well as changing the default configuration each time.
def run_smac(DATA_DIR, WORKING_DIR, domain_file, instance_dir, instances_with_features : dict, walltime_limit, n_trials, n_workers):
    ## Configuration Space ##
    ## Define parameters to select models
    os.mkdir(WORKING_DIR)

    alias = Categorical ('alias', ['lama-first'], default='lama-first')
    queue_type = Categorical("queue_type", ["trained", "roundrobintrained", "fifo"], default='trained')
    #,'noveltyfifo','roundrobinnovelty'], default='trained')

    parameters = [alias,queue_type]
    conditions = []

    ############################
    ### Gather sk_models
    #############################
    sk_models = [name for name in os.listdir(os.path.join(DATA_DIR,PARTIAL_GROUNDING_RULES_DIR)) if name.startswith(PREFIX_SK_MODELS)]
    sk_models_per_action_schema = defaultdict(lambda : ['none'])
    for model in sk_models:
        for n in os.listdir(os.path.join(DATA_DIR, PARTIAL_GROUNDING_RULES_DIR, model)):
            if n == 'relevant_rules':
                continue
            sk_models_per_action_schema[n].append(model)

    ############################
    ### Gather aleph probability_class, good_rule and bad_rule models
    #############################
    # aleph_model_filenames = [name for name in os.listdir(os.path.join(DATA_DIR, PARTIAL_GROUNDING_ALEPH_DIR)) if name.endswith(SUFFIX_ALEPH_MODELS)]
    # for model_filename in aleph_model_filenames:
    #     if 'class_probability' in model_filename:
    #         with open(os.path.join(DATA_DIR, PARTIAL_GROUNDING_ALEPH_DIR, model_flename)) as probability_model:
    #             for line in probability_model.readlines():
    #                 schema = line.split(":-")[0].strip()
    #                 sk_models_per_action_schema [schema].append(model_filename)


    ############################
    ### Create model parameters
    #############################

    for schema, models in sk_models_per_action_schema.items():
        m = Categorical(f"model_{schema}", models)
        parameters.append(m)
        conditions.append(InCondition(child=m, parent=queue_type, values=["trained", "roundrobintrained"]))


    ############################
    ### Stopping condition parameters
    #############################

    # TODO!!




    cs = ConfigurationSpace(seed=2023) # Fix seed for reproducibility
    cs.add_hyperparameters(parameters)
    cs.add_conditions(conditions)

    evaluator = Eval (DATA_DIR, WORKING_DIR, domain_file, instance_dir, sk_models_per_action_schema)


    print ([ins for ins in instances_with_features])
    print(instances_with_features)
    scenario = Scenario(
        configspace=cs, deterministic=True,
        output_directory=os.path.join(WORKING_DIR, 'smac'),
        walltime_limit=walltime_limit,
        n_trials=n_trials,
        n_workers=n_workers,
        instances=[ins for ins in instances_with_features],
        instance_features=instances_with_features
    )

    # Use SMAC to find the best configuration/hyperparameters
    smac = HyperparameterOptimizationFacade(scenario, evaluator.target_function)
    incumbent = smac.optimize()

    print("Chosen configuration: ", incumbent)
    if 'trained' in  incumbent['queue_type']:
        copy_model_to_folder(incumbent, sk_models_per_action_schema, DATA_DIR, os.path.join(WORKING_DIR, 'incumbent'), symlink=False )

    with open(os.path.join(WORKING_DIR, 'incumbent', 'config'), 'w') as config_file:
        json.dump(incumbent, config_file)
        #config_file.write(f"--alias {incumbent['alias']} --grounding-queue {incumbent['queue_type']}")
