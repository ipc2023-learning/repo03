#!/usr/bin/python
# -*- coding: utf-8 -*-

from sys import exit 
import pickle
import os.path

from subdominization.rule_evaluator import RulesEvaluator

from numpy import std
from collections import defaultdict



class TrainedModel():
    def __init__(self, modelFolder, task):
        if (not os.path.isdir(modelFolder)):
            exit("Error: given --trained-model-folder is not a folder: " + modelFolder)
        self.model = {}
        found_rules = False
        found_model = False
        for file in os.listdir(modelFolder):
            if (os.path.isfile(os.path.join(modelFolder, file))):
                if (file.endswith(".model")):
                    with open(os.path.join(modelFolder, file), "rb") as modelFile:
                        found_model = True
                        self.model[file[:-6]] = pickle.load(modelFile)
                elif (file == "relevant_rules"):
                    with open(os.path.join(modelFolder, file), "r") as rulesFile:
                        found_rules = True
                        self.ruleEvaluator = RulesEvaluator(rulesFile.readlines(), task)
        if (not found_rules):
            exit("Error: no relevant_rules file in " + modelFolder)
        if (not found_model):
            exit("Error: no *.model files in " + modelFolder)
        self.no_rule_schemas = set()
        self.estimates_per_schema = defaultdict(list)
        self.values_off_for_schema = set()
        
    def get_estimate(self, action):
        # returns the probability that the given action is part of a plan

        if (not action.predicate.name in self.model):
            self.no_rule_schemas.add(action.predicate.name)
            return None
        
        if (self.model[action.predicate.name].is_classifier):
            # the returned list has only one entry (estimates for the input action), 
            # of which the second entry is the probability that the action is in the plan (class 1)
            prob_estimation = self.model[action.predicate.name].model.predict_proba([self.ruleEvaluator.evaluate(action)])[0]

            if len(prob_estimation) > 1:
                assert len(prob_estimation) == 2
                estimate = prob_estimation[1]
            else:
                estimated_class = self.model[action.predicate.name].model.predict([self.ruleEvaluator.evaluate(action)])[0]
                if estimated_class == 1:
                    estimate = 1
                else:
                    assert estimated_class == 0
                    estimate = 0
        else:
            estimate = self.model[action.predicate.name].model.predict([self.ruleEvaluator.evaluate(action)])[0]

        if (estimate < 0 or estimate > 1): # in case the estimate is off
            self.values_off_for_schema.add(action.predicate.name)
            if (estimate < 0):
                estimate = 0
            else:
                estimate = 1
        
        self.estimates_per_schema[action.predicate.name].append(estimate) # TODO do we really need this? what's the memory overhead?
               
        return estimate
        
    def get_estimates(self, actions):
        # returns the probabilities that the given actions is part of a plan
        # all actions need to be from the same schema!
        
        schema = actions[0].predicate.name
        
        if (not schema in self.model):
            self.no_rule_schemas.add(schema)
            return None
        
        if (self.model[schema].is_classifier):
            # the returned list has only one entry (estimates for the input action), 
            # of which the second entry is the probability that the action is in the plan (class 1)
            prob_estimates = [p for p in self.model[schema].model.predict_proba([self.ruleEvaluator.evaluate(a) for a in actions])]
            if len(prob_estimates[0]) > 1:
                estimates = [p[1] for p in prob_estimates]
            else:
                value = self.model[schema].model.predict([self.ruleEvaluator.evaluate(actions[0])])[0]
                estimates = [value for a in actions]
        else:
            estimates = self.model[schema].model.predict([self.ruleEvaluator.evaluate(a) for a in actions])        
          
        if (any(e < 0 or e > 1 for e in estimates)):
            self.values_off_for_schema.add(schema)
            
        estimates = [min(max(e, 0), 1) for e in estimates]
        
        self.estimates_per_schema[schema] += estimates
            
        return estimates
    
    def print_stats(self):
        print("schema \t AVG \t STDDEV")
        for key, estimates in self.estimates_per_schema.items():
            print(key, sum(estimates) / len(estimates), std(estimates))
        for schema in self.no_rule_schemas:
            print("no relevant rule for action schema", schema)
        for schema in self.values_off_for_schema:
            print("bad estimate(s) for action schema", schema)

        


