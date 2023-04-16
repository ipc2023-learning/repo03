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

# from functools import partial

INTERMEDIATE_SMAC_MODELS = 'intermediate-smac-models'
PARTIAL_GROUNDING_RULES_DIR = 'partial-grounding-rules'

class Eval:
    def __init__(self, WORKING_DIR, domain_file, instances_dir, sk_models_per_action_schema):
        self.WORKING_DIR = os.path.abspath(WORKING_DIR)
        self.MY_DIR = os.path.dirname(os.path.realpath(__file__))
        self.sk_models_per_action_schema=sk_models_per_action_schema

        self.SMAC_MODELS_DIR = os.path.join(self.WORKING_DIR, INTERMEDIATE_SMAC_MODELS)
        if os.path.exists(self.SMAC_MODELS_DIR):
            shutil.rmtree(self.SMAC_MODELS_DIR)
        os.mkdir(self.SMAC_MODELS_DIR)
        self.instances_dir = instances_dir
        self.domain_file = domain_file


    def get_unique_model_name(self, config):
        print(config, [a for a in self.sk_models_per_action_schema])
        assert all([f'model_{aschema}' in config for aschema in self.sk_models_per_action_schema])
        return "-".join([str(opts.index(config[f'model_{aschema}'])) for aschema, opts in self.sk_models_per_action_schema.items()])


    def target_function (self, config: Configuration, instance: str, seed: int) -> float:
        # create folder for the model
        using_model = all ([f'model_{aschema}' in config for aschema in self.sk_models_per_action_schema]) and \
            any  ([config[f'model_{aschema}'] != 'none' for aschema in self.sk_models_per_action_schema])

        extra_parameters = ['--alias', config['alias'], '--grounding-queue', config['queue_type']]

        if using_model:
            config_name = self.get_unique_model_name(config)
            model_path = os.path.join(self.SMAC_MODELS_DIR, config_name)
            if not os.path.exists(model_path):
                os.mkdir(model_path)

                collected_relevant_rules = []
                for aschema in self.sk_models_per_action_schema:
                    if config[f'model_{aschema}'] == 'none':
                        continue
                    assert os.path.exists(os.path.join(self.WORKING_DIR,PARTIAL_GROUNDING_RULES_DIR, config[f'model_{aschema}'], aschema))
                    os.symlink(os.path.join(self.WORKING_DIR,PARTIAL_GROUNDING_RULES_DIR, config[f'model_{aschema}'], aschema), os.path.join(model_path, aschema))

                    with open(os.path.join(self.WORKING_DIR,PARTIAL_GROUNDING_RULES_DIR, config[f'model_{aschema}'], 'relevant_rules')) as rfile:
                        for line in rfile:
                            if line.startswith (aschema[:-6] + " ("):
                                collected_relevant_rules.append(line.strip())

                with open(os.path.join(model_path, 'relevant_rules'), 'w') as f:
                    f.write('\n'.join(collected_relevant_rules))

                # extra_parameters += ['--model', config_name]

        else:
            model_path = '.'

        instance_file = os.path.join(self.instances_dir, instance + ".pddl")
        assert(os.path.exists(instance_file))


        print (" ".join([sys.executable, f'{self.MY_DIR}/../plan-partial-grounding.py', model_path, self.domain_file, instance_file] + extra_parameters))

        Call([sys.executable, f'{self.MY_DIR}/../plan-partial-grounding.py', model_path, self.domain_file, instance_file] + extra_parameters, 'smac-plan').wait()

        # Go over configuration to create model
        # ./plan.py using config + model
        # print(self.WORKING_DIR)

        # Our objective function takes into consideration:
        # PAR10 score with respect to runtime
        # PAR10 score with respect to operators
        # PAR10 score with respect to quality

        return 10.0


# Note: default configuration should solve at least 50% of the instances. Pick instances
# with LAMA accordingly. If we run SMAC multiple times, we can use different instances
# set, as well as changing the default configuration each time.

def run_smac(WORKING_DIR, domain_file, instance_dir, instances_with_features : dict, walltime_limit, n_trials, n_workers):
    ## Configuration Space ##
    ## Define parameters to select models

    alias = Categorical ('alias', ['lama'], default='lama')
    queue_type = Categorical("queue_type", ["trained", "roundrobintrained",'noveltyfifo','roundrobinnovelty'], default='trained')

    parameters = [alias,queue_type]
    conditions = []

     # Gather model_names
    sk_models = [name for name in os.listdir(os.path.join(WORKING_DIR,PARTIAL_GROUNDING_RULES_DIR)) if name.startswith('model_')]

    sk_models_per_action_schema = defaultdict(lambda : ['none'])
    for model in sk_models:
        for n in os.listdir(os.path.join(WORKING_DIR, PARTIAL_GROUNDING_RULES_DIR, model)):
            if n == 'relevant_rules':
                continue
            sk_models_per_action_schema[n].append(model)

    for schema, models in sk_models_per_action_schema.items():
        m = Categorical(f"model_{schema}", models)
        parameters.append(m)
        conditions.append(InCondition(child=m, parent=queue_type, values=["trained", "roundrobintrained"]))


    cs = ConfigurationSpace(seed=2023) # Fix seed for reproducibility
    cs.add_hyperparameters(parameters)
    cs.add_conditions(conditions)

    evaluator = Eval (WORKING_DIR, domain_file, instance_dir, sk_models_per_action_schema)


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

    print(incumbent)
