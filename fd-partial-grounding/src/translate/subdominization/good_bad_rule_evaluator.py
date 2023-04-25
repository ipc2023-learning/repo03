#! /usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
import os.path

from .rule_evaluator import RulesEvaluator


class GodBadRuleEvaluator:
    def __init__(self, task, args):
        self.good_rule_evaluator = None
        self.bad_rule_evaluator = None
        if os.path.isfile(os.path.join(args.trained_model_folder, "good_rules.rules")):
            with open(os.path.join(args.trained_model_folder, "good_rules.rules")) as rules_file:
                self.good_rule_evaluator = RulesEvaluator(rules_file.readlines(), task)
        if os.path.isfile(os.path.join(args.trained_model_folder, "bad_rules.rules")):
            with open(os.path.join(args.trained_model_folder, "bad_rules.rules")) as rules_file:
                self.bad_rule_evaluator = RulesEvaluator(rules_file.readlines(), task)

        self.num_good_actions = defaultdict(int)
        self.num_bad_actions = defaultdict(int)

    def get_action_schemas(self):
        schemas = set()
        if self.good_rule_evaluator:
            schemas = set(self.good_rule_evaluator.get_action_schemas())
        if self.bad_rule_evaluator:
            schemas = schemas.union(set(self.bad_rule_evaluator.get_action_schemas()))
        return schemas

    def is_good_action(self, action):
        if self.good_rule_evaluator:
            is_good = any(e == 1 for e in self.good_rule_evaluator.evaluate(action))
            if is_good:
                self.num_good_actions[action.predicate.name] += 1
            return is_good
        else:
            return False

    def is_bad_action(self, action):
        if self.bad_rule_evaluator:
            is_bad = any(e == 1 for e in self.bad_rule_evaluator.evaluate(action))
            if is_bad:
                self.num_bad_actions[action.predicate.name] += 1
            return is_bad
        else:
            return False

    def print_stats(self):
        for schema, num in self.num_good_actions.items():
            print(f"Detected {num} good operators of action schema {schema}.")
        for schema, num in self.num_bad_actions.items():
            print(f"Detected {num} bad operators of action schema {schema}.")
