from rule_evaluator import *

from collections import defaultdict

class TrainingRule:
    def __init__(self, rules_text, rules = [], count0 = 0, count1 = 0):
        self.rules_text = rules_text
        self.rules = rules
        self.evaluation_result_count_0 = count0
        self.evaluation_result_count_1 = count1
        self.single_rule_relevant = (len(self.rules_text) == 1) and self.evaluation_result_count_0 > 0 and self.evaluation_result_count_1 > 0

    def load(self, task, max_training_examples):
        if not self.single_rule_relevant and self.evaluation_result_count_0 + self.evaluation_result_count_1 < max_training_examples :
            self.rules = [RuleEval (l, task) for l in self.rules_text]
        else:
            self.rules = None

    def evaluate(self,  arguments):
        if not self.rules:
            return 0
        indexes_1 = [i for i, r in enumerate(self.rules) if r.evaluate(arguments) == 1]
        if len(indexes_1) == 0:
            self.evaluation_result_count_0 += 1
        elif len(indexes_1) == len(self.rules):
            self.evaluation_result_count_1 += 1
        else:
            rules_text_1 = [self.rules_text[i] for i in indexes_1]
            rules_1 = [self.rules[i] for i in indexes_1]
            res = TrainingRule(rules_text_1, rules_1, self.evaluation_result_count_0, self.evaluation_result_count_1 + 1)

            self.rules_text = [self.rules_text[i] for i in range(len(self.rules)) if i not in indexes_1]
            self.rules = [self.rules[i] for i in range(len(self.rules)) if i not in indexes_1]
            self.evaluation_result_count_0 += 1

            return res
        return None

    def get_text(self):
        return self.rules_text[0]
        
class RuleTrainingEvaluator:
    def __init__(self, rules_text):
        self.rules = {}
        rules_per_schema = defaultdict(list)
        for l in rules_text:
            action_schema = l.split(" (")[0]
            rules_per_schema[action_schema].append(l)

        for schema in rules_per_schema:
            self.rules[schema] = [TrainingRule(rules_per_schema[schema])]
            
    def init_task (self, task, max_training_examples):
        for schema, rs in self.rules.items():
            for r in rs:
                r.load(task, max_training_examples)
            
    def evaluate(self, action):
        name, arguments = action.split("(")
        arguments = list(map(lambda x: x.strip(), arguments.strip()[:-1].split(",")))

        if name in self.rules:
            new_rules = [rule.evaluate(arguments) for rule in self.rules[name]]
            self.rules[name] += [r for r in new_rules if r]
        ################################# This case can happen if we do not have any rules for some schemas
        # else:
        #     print ("Error: unrecognized action name: {}".format(name))
        #     exit()
    def get_relevant_rules(self):
        return [rule.get_text() for (schema, rules)  in self.rules.items() for rule in rules if rule.evaluation_result_count_0 > 0 and rule.evaluation_result_count_1 > 0]

    def print_statistics(self):
        for schema, rs in self.rules.items():
            for r in rs:
                print(r.evaluation_result_count_0, r.evaluation_result_count_1, r.rules_text)

