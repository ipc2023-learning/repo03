from ConfigSpace import Categorical, Float, Configuration, ConfigurationSpace, InCondition
from smac import HyperparameterOptimizationFacade, Scenario

from lab.calls.call import Call

import sys
import os
import json
import subprocess
import re
import shutil

from collections import defaultdict


# from functools import partial

# Hardcoded paths that depend on the trraining part. This could be passed by parameter instead
PARTIAL_GROUNDING_RULES_DIR = 'partial-grounding-rules'
PARTIAL_GROUNDING_ALEPH_DIR  = 'partial-grounding-aleph'
SUFFIX_ALEPH_MODELS = '.rules'
PREFIX_SK_MODELS = 'model_'

# Hardcoded paths
INTERMEDIATE_SMAC_MODELS = 'intermediate-smac-models'


class Eval:
    def __init__(self, DATA_DIR, WORKING_DIR, domain_file, instances_dir, candidate_models):
        self.DATA_DIR = DATA_DIR
        self.MY_DIR = os.path.dirname(os.path.realpath(__file__))
        self.candidate_models=candidate_models

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



class CandidateModels:
    def __init__(self):
        self.sk_models_per_action_schema = defaultdict(lambda : ['none'])
        self.good_rules = []
        self.bad_rules = []

    def is_using_model(self, config):
        return all ([f'model_{aschema}' in config for aschema in self.sk_models_per_action_schema]) and \
            any  ([config[f'model_{aschema}'] != 'none' for aschema in self.sk_models_per_action_schema])

    def get_unique_model_name(self, config):
        assert all([f'model_{aschema}' in config for aschema in self.sk_models_per_action_schema])
        prefix = lambda x : "sk" if x.startswith(PREFIX_SK_MODELS) else ("a" if x.endswith(SUFFIX_ALEPH_MODELS) else "")
        return "-".join([prefix(config[f'model_{aschema}']) + str(opts.index(config[f'model_{aschema}'])) for aschema, opts in self.sk_models_per_action_schema.items()])


    def load_sk_folder(self, sk_folder):
        self.sk_folder = sk_folder

        sk_models = [name for name in os.listdir(sk_folder) if name.startswith(PREFIX_SK_MODELS)]

        for model in sk_models:
            for n in os.listdir(os.path.join(sk_folder, model)):
                if n == 'relevant_rules':
                    continue
                self.sk_models_per_action_schema[n[:-6]].append(model)


    def load_aleph_folder(self, aleph_folder):
        self.aleph_folder = aleph_folder

        aleph_model_filenames = [name for name in os.listdir(aleph_folder) if name.endswith(SUFFIX_ALEPH_MODELS)]
        for model_filename in aleph_model_filenames:
            with open(os.path.join(aleph_folder, model_filename)) as model_file:
                if 'class_probability' in model_filename:
                    for line in model_file.readlines():
                        schema = line.split(":-")[ 0].strip()
                        self.sk_models_per_action_schema [schema].append(model_filename)
                elif 'good_rules' in model_filename:
                    self.good_rules += model_file.readlines()
                elif 'bad_rules' in model_filename:
                    self.bad_rules += model_file.readlines()
                else:
                    print (f"Warning: ignoring file of unknown type: {model_filename}")


    def copy_model_to_folder(self, config, target_dir, symlink=False ):
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        collected_relevant_rules = []
        collected_aleph_models = []
        for aschema in self.sk_models_per_action_schema:
            if config[f'model_{aschema}'].startswith(PREFIX_SK_MODELS):
                model_file = os.path.join(self.sk_folder, config[f'model_{aschema}'], aschema + ".model")
                target_file = os.path.join(target_dir, aschema + ".model")
                assert os.path.exists(model_file)
                if symlink:
                    os.symlink(model_file, target_file)
                else:
                    shutil.copy(model_file, target_file)

                with open(os.path.join(self.sk_folder, config[f'model_{aschema}'], 'relevant_rules')) as rfile:
                    for line in rfile:
                        if line.startswith (aschema + " ("):
                            collected_relevant_rules.append(line.strip())
            elif config[f'model_{aschema}'].endswith(SUFFIX_ALEPH_MODELS):
                with open(os.path.join(self.aleph_folder, config[f'model_{aschema}'])) as probability_model:
                    for line in probability_model.readlines():
                        schema = line.split(":-")[0].strip()
                        if schema == aschema:
                            collected_aleph_models.append(line)

            else:
                assert config[f'model_{aschema}'] == 'none'

        if collected_relevant_rules:
            with open(os.path.join(target_dir, 'relevant_rules'), 'w') as f:
                f.write('\n'.join(collected_relevant_rules))

        if collected_aleph_models:
            with open(os.path.join(target_dir, 'probability_class.rules'), 'w') as f:
                f.write('\n'.join(collected_aleph_models))



# Note: default configuration should solve at least 50% of the instances. Pick instances
# with LAMA accordingly. If we run SMAC multiple times, we can use different instances
# set, as well as changing the default configuration each time.
def run_smac(DATA_DIR, WORKING_DIR, domain_file, instance_dir, instances_with_features : dict, walltime_limit, n_trials, n_workers):
    DATA_DIR = os.path.abspath(DATA_DIR) # Make sure path is absolute so that symlinks work

    ## Configuration Space ##
    ## Define parameters to select models
    os.mkdir(WORKING_DIR)

    ############################
    ### Gather sk_models,  aleph probability_class, good_rule and bad_rule models
    #############################
    candidate_models = CandidateModels()
    candidate_models.load_sk_folder(os.path.join(DATA_DIR,PARTIAL_GROUNDING_RULES_DIR))
    candidate_models.load_aleph_folder(os.path.join(DATA_DIR, PARTIAL_GROUNDING_ALEPH_DIR))

    ############################
    ### Create model parameters
    #############################

    alias = Categorical ('alias', ['lama-first'], default='lama-first')
    queue_type = Categorical("queue_type", ["trained", "roundrobintrained", "fifo", "lifo"], default='trained')

    parameters = [alias,queue_type]
    conditions = []

    for schema, models in candidate_models.sk_models_per_action_schema.items():
        m = Categorical(f"model_{schema}", models)
        parameters.append(m)
        conditions.append(InCondition(child=m, parent=queue_type, values=["trained", "roundrobintrained"]))

    for i, r in enumerate(candidate_models.good_rules):
        parameters.append(Categorical(f"good_{i}", [False, True]))

    for i, r in enumerate(candidate_models.bad_rules):
        parameters.append(Categorical(f"bad{i}", [False, True]))

    ############################
    ### Stopping condition parameters
    #############################

    # TODO!!




    cs = ConfigurationSpace(seed=2023) # Fix seed for reproducibility
    cs.add_hyperparameters(parameters)
    cs.add_conditions(conditions)

    evaluator = Eval (DATA_DIR, WORKING_DIR, domain_file, instance_dir, candidate_models)


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
        candidate_models.copy_model_to_folder(incumbent, os.path.join(WORKING_DIR, 'incumbent'), symlink=False )
    else:
        os.mkdir(os.path.join(WORKING_DIR, 'incumbent'))

    with open(os.path.join(WORKING_DIR, 'incumbent', 'config'), 'w') as config_file:
        json.dump(incumbent.get_dictionary(), config_file)
        #config_file.write(f"--alias {incumbent['alias']} --grounding-queue {incumbent['queue_type']}")
