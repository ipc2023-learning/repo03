#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import os.path
import pickle
from sys import exit

from subdominization.rule_evaluator import RulesEvaluator
from subdominization.rule_evaluator_aleph import RuleEvaluatorAleph


class TrainedModel:
    def __init__(self, model_folder, task):
        if not os.path.isdir(model_folder):
            exit("Error: given --trained-model-folder is not a folder: " + model_folder)
        self.model = {}
        found_rules = False
        found_model = False
        for file in os.listdir(model_folder):
            if os.path.isfile(os.path.join(model_folder, file)):
                if file.endswith(".model"):
                    with open(os.path.join(model_folder, file), "rb") as modelFile:
                        found_model = True
                        self.model[file[:-6]] = pickle.load(modelFile)
                elif file == "relevant_rules":
                    with open(os.path.join(model_folder, file), "r") as rulesFile:
                        found_rules = True
                        self.ruleEvaluator = RulesEvaluator(rulesFile.readlines(), task)
        if not found_rules:
            exit("Error: no relevant_rules file in " + model_folder)
        if not found_model:
            exit("Error: no *.model files in " + model_folder)

        self.no_rule_schemas = set()

        # statistics
        self.min_estimate_by_schema = {schema: math.inf for schema in self.model.keys()}
        self.max_estimate_by_schema = {schema: 0 for schema in self.model.keys()}
        self.sum_estimates_by_schema = {schema: 0 for schema in self.model.keys()}
        self.num_estimates_by_schema = {schema: 0 for schema in self.model.keys()}
        self.values_off_for_schema = set()

    def get_trained_schemas(self):
        return self.model.keys()
        
    def get_estimate(self, action):
        # returns the probability that the given action is part of a plan

        schema = action.predicate.name

        if schema not in self.model:
            self.no_rule_schemas.add(schema)
            return None
        
        if self.model[schema].is_classifier:
            # the returned list has only one entry (estimates for the input action), 
            # of which the second entry is the probability that the action is in the plan (class 1)
            prob_estimation = self.model[schema].model.predict_proba([self.ruleEvaluator.evaluate(action)])[0]

            if len(prob_estimation) > 1:
                assert len(prob_estimation) == 2
                estimate = prob_estimation[1]
            else:
                estimated_class = self.model[schema].model.predict([self.ruleEvaluator.evaluate(action)])[0]
                if estimated_class == 1:
                    estimate = 1
                else:
                    assert estimated_class == 0
                    estimate = 0
        else:
            estimate = self.model[schema].model.predict([self.ruleEvaluator.evaluate(action)])[0]

        if estimate < 0 or estimate > 1:  # in case the estimate is off
            self.values_off_for_schema.add(schema)
            if estimate < 0:
                estimate = 0
            else:
                estimate = 1

        self.min_estimate_by_schema[schema] = min(estimate, self.min_estimate_by_schema[schema])
        self.max_estimate_by_schema[schema] = max(estimate, self.max_estimate_by_schema[schema])
        self.sum_estimates_by_schema[schema] += estimate
        self.num_estimates_by_schema[schema] += 1
               
        return estimate
        
    def get_estimates(self, actions):
        # returns the probabilities that the given actions is part of a plan
        # all actions need to be from the same schema!
        
        schema = actions[0].predicate.name
        
        if schema not in self.model:
            self.no_rule_schemas.add(schema)
            return None
        
        if self.model[schema].is_classifier:
            # the returned list has only one entry (estimates for the input action), 
            # of which the second entry is the probability that the action is in the plan (class 1)
            prob_estimates = [p for p in self.model[schema].model.predict_proba([self.ruleEvaluator.evaluate(a) for a in actions])]
            if len(prob_estimates[0]) > 1:
                estimates = [p[1] for p in prob_estimates]
            else:
                value = self.model[schema].model.predict([self.ruleEvaluator.evaluate(actions[0])])[0]
                estimates = [value for _ in actions]
        else:
            estimates = self.model[schema].model.predict([self.ruleEvaluator.evaluate(a) for a in actions])        
          
        if any(e < 0 or e > 1 for e in estimates):
            self.values_off_for_schema.add(schema)
            
        estimates = [min(max(e, 0), 1) for e in estimates]

        self.min_estimate_by_schema[schema] = min(min(estimates), self.min_estimate_by_schema[schema])
        self.max_estimate_by_schema[schema] = max(max(estimates), self.max_estimate_by_schema[schema])
        self.sum_estimates_by_schema[schema] += sum(estimates)
        self.num_estimates_by_schema[schema] += len(estimates)
            
        return estimates
    
    def print_stats(self):
        print("Statistics of rule-based model:")
        print(f"schema{' ' * 9} \t #predictions \t min \t avg \t max")
        for schema in self.model.keys():
            avg = 0
            if self.num_estimates_by_schema[schema] > 0:
                avg = round(self.sum_estimates_by_schema[schema] / self.num_estimates_by_schema[schema], 2)
            print(f"{schema}{' ' * max(0, 15 - len(schema))} \t "
                  f"{self.num_estimates_by_schema[schema]} \t\t "
                  f"{round(self.min_estimate_by_schema[schema], 2)} \t "
                  f"{avg} \t "
                  f"{round(self.max_estimate_by_schema[schema], 2)}")
        for schema in self.no_rule_schemas:
            print("no model for action schema", schema)
        for schema in self.values_off_for_schema:
            print("bad estimate(s) for action schema", schema)


class HybridModel:
    def __init__(self, model_folder, task):
        if not os.path.isdir(model_folder):
            exit("Error: given --trained-model-folder is not a folder: " + model_folder)
        self.model = {}
        self.rule_model = None
        self.aleph_model = None
        has_relevant_rules_file = False
        has_model_file = False
        for file in os.listdir(model_folder):
            if os.path.isfile(os.path.join(model_folder, file)):
                if file.endswith(".model"):
                    has_model_file = True
                elif file == "relevant_rules":
                    has_relevant_rules_file = True
                elif file == "class_probability.rules":
                    with open(os.path.join(model_folder, file), "r") as aleph_rules:
                        self.aleph_model = RuleEvaluatorAleph(aleph_rules.readlines(), task)
                        for schema in self.aleph_model.get_trained_schemas():
                            self.model[schema] = self.aleph_model
        if has_relevant_rules_file and has_model_file:
            self.rule_model = TrainedModel(model_folder, task)
            for schema in self.rule_model.get_trained_schemas():
                if schema in self.model:
                    exit(f"ERROR: already have an aleph model for schema {schema}")
                self.model[schema] = self.rule_model
        elif has_relevant_rules_file and not has_model_file:
            exit("ERROR: found relevant_rules file but no *.model files in " + model_folder)
        elif has_model_file and not has_relevant_rules_file:
            exit("ERROR: found *.model files but no relevant_rules file in " + model_folder)

        self.no_rule_schemas = set()

    def get_trained_schemas(self):
        return self.model.keys()

    def get_estimate(self, action):
        # returns the probabilities that the given actions is part of a plan
        # all actions need to be from the same schema!

        schema = action.predicate.name

        if schema not in self.model:
            self.no_rule_schemas.add(schema)
            return None

        return self.model[schema].get_estimate(action)

    def get_estimates(self, actions):
        # returns the probabilities that the given actions is part of a plan
        # all actions need to be from the same schema!

        schema = actions[0].predicate.name

        # assert all(schema == a.predicate.name for a in actions)

        if schema not in self.model:
            self.no_rule_schemas.add(schema)
            return None

        return self.model[schema].get_estimates(actions)

    def print_stats(self):
        if self.rule_model and self.rule_model.get_trained_schemas():
            self.rule_model.print_stats()
        else:
            print("No rule model.")
        if self.aleph_model and self.aleph_model.get_trained_schemas():
            self.aleph_model.print_stats()
        else:
            print("No aleph model.")

