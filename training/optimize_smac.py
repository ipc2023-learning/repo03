from ConfigSpace import Constant, Categorical, Float, Configuration, ConfigurationSpace, InCondition
from smac import AlgorithmConfigurationFacade, Scenario, HyperparameterOptimizationFacade

from smac.initial_design.default_design import DefaultInitialDesign
from smac.utils.configspace import get_config_hash
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

# Hardcoded paths
INTERMEDIATE_SMAC_MODELS = 'intermediate-smac-models'

class Eval:
    def __init__(self, DATA_DIR, WORKING_DIR, domain_file, instances_dir, candidate_models, trial_walltime_limit, instances_properties):
        self.DATA_DIR = DATA_DIR
        self.MY_DIR = os.path.dirname(os.path.realpath(__file__))
        self.candidate_models=candidate_models
        self.trial_walltime_limit=trial_walltime_limit

        self.SMAC_MODELS_DIR = os.path.abspath(os.path.join(WORKING_DIR, INTERMEDIATE_SMAC_MODELS))
        if os.path.exists(self.SMAC_MODELS_DIR):
            shutil.rmtree(self.SMAC_MODELS_DIR)
        os.mkdir(self.SMAC_MODELS_DIR)
        self.RUNNING_DIR = os.path.abspath(os.path.join(WORKING_DIR, 'runs'))
        os.mkdir(self.RUNNING_DIR)
        self.instances_dir = os.path.abspath(instances_dir)
        self.domain_file = domain_file

        self.regex_total_time = re.compile(rb"INFO\s+Planner time:\s(.+)s", re.MULTILINE)
        self.regex_operators = re.compile(rb"Translator operators:\s(.+)", re.MULTILINE)
        self.regex_plan_cost = re.compile(rb"\[t=.*s, .* KB\] Plan cost:\s(.+)\n", re.MULTILINE)
        self.regex_no_solution = re.compile(rb"\[t=.*KB\] Completely explored state space.*no solution.*", re.MULTILINE)
        self.instances_properties = instances_properties

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

        extra_parameters = ['--h2-preprocessor', '--alias', config['alias'], '--grounding-queue', config['queue_type'],
                            '--incremental-grounding', '--incremental-grounding-increment-percentage', '20',
                            '--termination-condition', config['termination-condition']]


        instance_file = os.path.join(self.instances_dir, instance + ".pddl")
        assert(os.path.exists(instance_file))

        command=[sys.executable, f'{self.MY_DIR}/../plan-partial-grounding.py', model_path, self.domain_file, instance_file] + extra_parameters
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        time_limit = self.trial_walltime_limit

        try:
            output, error_output = proc.communicate(timeout=time_limit)

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
            proc.kill()
            print (f"WARNING: Command failed: {' '.join(command)}")
            print (f"Ran {instance} with queue {config['queue_type']} and model {config_name}: not solved due to crash")
            return 10000000

        except subprocess.TimeoutExpired as e:
            proc.kill()
            print (f"Ran {instance} with queue {config['queue_type']} and model {config_name}: not solved due to time limit")
            # print(e)
            return 10000000

        except:
            proc.kill()
            print (f"Error: Command failed: {' '.join(command)}")
            return 10000000


    def target_function_bad_rules (self, config: Configuration, instance: str, seed: int) -> float:
        if not self.candidate_models.is_using_model(config):
            num_operators = self.instances_properties[instance]['translator_operators']
            coverage = self.instances_properties[instance]['coverage']
            print (f"Ran {instance} without bad rules: full grounding size is {num_operators}, was solved by the baseline {coverage}")
            return num_operators# + self.candidate_models.total_bad_rules() if coverage else 10000000

        # if self.instances_properties[instance]['coverage']:
        #     print(f"Starting with {instance} solved by the baseline in {self.instances_properties[instance]['planner_wall_clock_time']}")
        # else:
        #     print(f"Starting with {instance}, not solved by the baseline")

        config_name = self.candidate_models.get_unique_model_name(config)
        num_bad_rules = self.candidate_models.num_bad_rules(config)
        model_path = os.path.join(self.SMAC_MODELS_DIR, config_name)
        if not os.path.exists(model_path):
            self.candidate_models.copy_model_to_folder(config, model_path, symlink=True)


        config_hash = get_config_hash(config)
        run_dir = os.path.join(self.RUNNING_DIR, "-".join([config_hash, instance]))

        os.mkdir(run_dir)
        os.chdir(run_dir)

        extra_parameters = ['--h2-preprocessor', '--alias', config['alias'], '--grounding-queue', config['queue_type'],
                            '--termination-condition', config['termination-condition'], '--ignore-bad-actions']

        instance_file = os.path.join(self.instances_dir, instance + ".pddl")
        assert(os.path.exists(instance_file))

        command=[sys.executable, f'{self.MY_DIR}/../plan-partial-grounding.py', model_path, self.domain_file, instance_file] + extra_parameters
        #print (" ".join(command[1:]))
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


        time_limit = self.trial_walltime_limit
        if self.instances_properties[instance]['coverage'] and self.instances_properties[instance]['planner_wall_clock_time'] < time_limit:
            # Give at least 5 seconds or 3 times the amount of the baseline
            time_limit = min(time_limit, max(self.instances_properties[instance]['planner_wall_clock_time']*3, 5 ))

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
                return num_operators #+ self.candidate_models.total_bad_rules() - num_bad_rules

            elif self.regex_no_solution.search(output):
                print (f"Ran {instance} with queue {config['queue_type']} and model {config_name}: not solved due to partial grounding")
                return 10000000
            else:
                print (f"WARNING: Ran {instance} with queue {config['queue_type']} and model {config_name}: not solved due to unknown reasons")
                print("Output: ", output.decode())
                if error_output:
                    print("Error Output: ", error_output.decode())
                return 10000000

        except subprocess.CalledProcessError:
            proc.kill()
            print (f"WARNING: Command failed: {' '.join(command)}")
            print (f"Ran {instance} with queue {config['queue_type']} and model {config_name}: not solved due to crash")
            return 10000000

        except subprocess.TimeoutExpired as e:
            proc.kill()
            print (f"Ran {instance} with queue {config['queue_type']} and model {config_name}: not solved due to time limit")
            return 10000000

        except:
            proc.kill()
            print (f"Error: Command failed: {' '.join(command)}")
            return 10000000



def run_smac_bad_rules(DATA_DIR, WORKING_DIR,
                       domain_file,
                       instance_dir, instances_with_features : dict, instances_properties : dict,
                       walltime_limit, trial_walltime_limit, n_trials,
                       n_workers,
                       PARTIAL_GROUNDING_ALEPH_DIRS  = ['partial-grounding-good-rules', 'partial-grounding-bad-rules']):

    DATA_DIR = os.path.abspath(DATA_DIR) # Make sure path is absolute so that symlinks work
    WORKING_DIR = os.path.abspath(WORKING_DIR) # Making path absolute for using SMAC with multiple cores
    os.mkdir(WORKING_DIR)
    cwd = os.getcwd() # Save current directory to restore it after SMAC optimization

    ## Configuration Space ##
    ## Define parameters to select models
    ############################
    ### Gather sk_models,  aleph probability_class, good_rule and bad_rule models
    #############################
    candidate_models = CandidateModels()
    for ALEPH_DIR in PARTIAL_GROUNDING_ALEPH_DIRS:
        candidate_models.load_aleph_folder(os.path.join(DATA_DIR, ALEPH_DIR))

    ############################
    ### Create model parameters
    #############################

    stopping_condition = Constant(f"termination-condition", 'full')
    alias = Constant('alias', 'lama-first')
    queue_type = Constant("queue_type", "ipc23-single-queue")

    parameters = [alias, queue_type, stopping_condition]
    conditions = []

    for i, r in enumerate(candidate_models.good_rules):
        parameters.append(Constant(f"good{i}", 1))

    for i, r in enumerate(candidate_models.bad_rules):
        parameters.append(Categorical(f"bad{i}", [False, True], default=True))

    cs = ConfigurationSpace(seed=2023) # Fix seed for reproducibility
    cs.add_hyperparameters(parameters)
    cs.add_conditions(conditions)

    evaluator = Eval (DATA_DIR, WORKING_DIR, domain_file, instance_dir, candidate_models, trial_walltime_limit, instances_properties)

    sorted_instances = sorted ([ins for ins in instances_with_features], key=lambda x : instances_with_features[x])

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
    smac = AlgorithmConfigurationFacade(scenario, evaluator.target_function_bad_rules)#cs.get_default_configuration())

    #smac = HyperparameterOptimizationFacade(scenario, evaluator.target_function_bad_rules,)

    incumbent_config = smac.optimize()


    # default_cost = smac.validate(cs.get_default_configuration())
    # print(f"Default cost: {default_cost}")
    # incumbent_cost = smac.validate(incumbent_config)
    # print(f"Incumbent cost: {incumbent_cost}")

    os.chdir(cwd)

    print("Chosen set of bad rules: ", incumbent_config)
    candidate_models.copy_model_to_folder(incumbent_config, os.path.join(WORKING_DIR, 'incumbent'), symlink=False )

    with open(os.path.join(WORKING_DIR, 'incumbent', 'config'), 'w') as config_file:
        properties = {k : v for k,v in incumbent_config.get_dictionary().items() if not k.startswith('good') and not k.startswith('bad')}
        properties['num_bad_rules'] = sum([1 for i, _ in enumerate(candidate_models.bad_rules) if [incumbent_config[f"bad{i}"]]])
        properties['num_good_rules'] = sum([1 for i, _ in enumerate(candidate_models.good_rules) if [incumbent_config[f"good{i}"]]])
        json.dump(properties, config_file)




# Note: default configuration should solve at least 50% of the instances. Pick instances
# with LAMA accordingly. If we run SMAC multiple times, we can use different instances
# set, as well as changing the default configuration each time.
def run_smac_partial_grounding(DATA_DIR, WORKING_DIR, domain_file, instance_dir, instances_with_features : dict, walltime_limit, trial_walltime_limit, n_trials, n_workers,
                               PARTIAL_GROUNDING_RULES_DIR = 'partial-grounding-sklearn',
                               PARTIAL_GROUNDING_ALEPH_DIRS  = ['partial-grounding-aleph', 'partial-grounding-good-rules', 'partial-grounding-bad-rules']
    ):
    DATA_DIR = os.path.abspath(DATA_DIR) # Make sure path is absolute so that symlinks work
    WORKING_DIR = os.path.abspath(WORKING_DIR) # Making path absolute for using SMAC with multiple cores

    cwd = os.getcwd() # Save current directory to restore it after SMAC optimization
    ## Configuration Space ##
    ## Define parameters to select models
    os.mkdir(WORKING_DIR)

    ############################
    ### Gather sk_models,  aleph probability_class, good_rule and bad_rule models
    #############################
    candidate_models = CandidateModels()
    candidate_models.load_sk_folder(os.path.join(DATA_DIR,PARTIAL_GROUNDING_RULES_DIR))
    for ALEPH_DIR in PARTIAL_GROUNDING_ALEPH_DIRS:
        candidate_models.load_aleph_folder(os.path.join(DATA_DIR, ALEPH_DIR))

    ############################
    ### Create model parameters
    #############################

    stopping_condition = Categorical(f"termination-condition", ['full', "relaxed", "relaxed5", "relaxed10", "relaxed20"])
    alias = Categorical('alias', ['lama-first'] + [f"seq-sat-fdss-2018-{i}" for i in range(0, 41)], default='lama-first')
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
        parameters.append(Constant(f"good{i}", 1))

    for i, r in enumerate(candidate_models.bad_rules):
        parameters.append(Constant(f"bad{i}",1 ))

    cs = ConfigurationSpace(seed=2023) # Fix seed for reproducibility
    cs.add_hyperparameters(parameters)
    cs.add_conditions(conditions)

    evaluator = Eval (DATA_DIR, WORKING_DIR, domain_file, instance_dir, candidate_models, trial_walltime_limit, None)

    sorted_instances = sorted ([ins for ins in instances_with_features], key=lambda x : instances_with_features[x])

    # for a in sorted_instances:
    #      print(a, instances_with_features[a] )

    scenario = Scenario(
        configspace=cs, deterministic=True,
        output_directory=PosixPath(os.path.join(WORKING_DIR, 'smac')),
        walltime_limit=walltime_limit,
        n_trials=n_trials,
        n_workers=n_workers,
        instances=sorted_instances,
        instance_features=instances_with_features,
        min_budget=len(sorted_instances)
    )

    # Use SMAC to find the best configuration/hyperparameters
    #smac = AlgorithmConfigurationFacade(scenario, evaluator.target_function) ,#initial_design=DefaultInitialDesign(scenario))
    smac = HyperparameterOptimizationFacade(scenario, evaluator.target_function,)

    incumbent_config = smac.optimize()

    os.chdir(cwd)


    print("Chosen configuration: ", incumbent_config)
    if 'ipc23' in incumbent_config['queue_type']:
        candidate_models.copy_model_to_folder(incumbent_config, os.path.join(WORKING_DIR, 'incumbent'), symlink=False )
    else:
        os.mkdir(os.path.join(WORKING_DIR, 'incumbent'))

    with open(os.path.join(WORKING_DIR, 'incumbent', 'config'), 'w') as config_file:
        properties = {k : v for k,v in incumbent_config.get_dictionary().items() if not k.startswith('good') and not k.startswith('bad')}

        properties['num_bad_rules'] = sum([1 for i, _ in enumerate(candidate_models.bad_rules) if [incumbent_config[f"bad{i}"]]])
        properties['num_good_rules'] = sum([1 for i, _ in enumerate(candidate_models.good_rules) if [incumbent_config[f"good{i}"]]])
        json.dump(properties, config_file)
