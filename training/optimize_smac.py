from ConfigSpace import Configuration, ConfigurationSpace

import numpy as np
from smac import HyperparameterOptimizationFacade, Scenario
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


def train(config: Configuration, instance: str, seed: int = 0) -> float:
    # Launch planner on multiple instances and try to solve them

    # create folder for the model
    # Go over configuration to create model
    # ./plan.py using config + model





    # Our objective function takes into consideration:
    # PAR10 score with respect to runtime
    # PAR10 score with respect to operators
    # PAR10 score with respect to quality

    return 1 - np.mean(scores)


# Note: default configuration should solve at least 50% of the instances. Pick instances
# with LAMA accordingly. If we run SMAC multiple times, we can use different instances
# set, as well as changing the default configuration each time.

def run_smac(WORKING_DIR, instance_set, walltime_limit, n_trials, n_workers):
    ## Configuration Space ##
    ## Define parameters to select models

    parameters = []

    for schema in action_schemas:
        # Gather model_names
        parameters.append(Categorical("model_{schema}", model_names))
        # parameters.append(Categorical("queue_type", ["simple", "round_robin"])) # TODO set these parameters better (e.g. add weighted round robin?)


    configspace = ConfigurationSpace(seed=2023) # Fix seed for reproducibility
    configspace.add_hyperparameters(parameters)

    scenario = Scenario(
        configspace=cs, deterministic=True,
        output_directory=WORKING_DIR,
        walltime_limit=walltime_limit,
        n_trials=n_trials,
        n_workers=n_workers,
        instances=instance_set
    )


    # Use SMAC to find the best configuration/hyperparameters
    smac = HyperparameterOptimizationFacade(scenario, train)
    incumbent = smac.optimize()
