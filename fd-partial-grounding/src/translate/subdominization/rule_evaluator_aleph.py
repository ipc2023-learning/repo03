#! /usr/bin/env python
# -*- coding: utf-8 -*-

import math
from collections import defaultdict
import pddl

import re
regexp_is_float = re.compile("\d.\d+")
def is_float(text):
    return regexp_is_float.match(text.strip())


def evaluate_inigoal_rule(rule, fact_list):
    def eval_constants(fact, constants):
        for (i, val) in constants:
            if fact.args[i] != val:
                return False
        return True
    compliant_values = set()

    predicate_name, arguments = rule.split("(")
    arguments = arguments.replace(")", "").replace("\n", "").replace(".", "").replace(" ", "").split(",")
    valid_arguments = tuple(set([a for a in arguments if a.startswith("?")]))
    constants = [(i, val) for (i, val) in enumerate(arguments) if val != "_" and not val.startswith("?")]
    positions_argument = {}

    for a in valid_arguments:
        positions_argument[a] = [i for (i, v) in enumerate(arguments) if v == a]

    arguments = valid_arguments
    for fact in fact_list:
        if type(fact) != pddl.Assign and fact.predicate == predicate_name and eval_constants(fact, constants):
            values = []
            for a in arguments:
                if len(set([fact.args[p] for p in positions_argument[a]])) > 1:
                    break
                values.append(fact.args[positions_argument[a][0]])

            if len(values) == len(arguments):
                compliant_values.add(tuple(values))

    return arguments, compliant_values


class AlephRule:
    def __init__(self, rule, task):
        self.rule_text = rule
        rule_type, rule = rule.split(":")
        rule_type = rule_type.strip()

        if rule_type == "ini":
            arguments, compliant_values = evaluate_inigoal_rule (rule, task.init)                        
        elif rule_type == "goal":
            arguments, compliant_values = evaluate_inigoal_rule (rule, task.goal.parts)                
        elif rule_type == "equal":
            arguments = tuple(rule[1:rule.find(')')].split(", "))
            compliant_values = set()
            accepted_types = set()
            action_schema = list(filter(lambda a : a.name == self.action_schema, task.actions))[0]
            argument_types = set([p.type_name for p in action_schema.parameters if p.name in arguments])

        self.action_arguments = []
        self.input_free_variables = []
        self.output_free_variables = []

        input_pos_arg = {}
        output_pos_arg = {}

        for i, arg in enumerate(arguments):
            if arg.startswith("?arg"):
                arg_id = int(arg[4:])
                self.action_arguments.append(arg_id)
                input_pos_arg [arg_id] = i
            else:
                assert(arg.startswith("?fv"))
                if arg.endswith("-io"):
                    self.input_free_variables.append(arg[:-3])
                    self.output_free_variables.append(arg[:-3])
                    input_pos_arg[arg[:-3]] = i
                    output_pos_arg[arg[:-3]] = i
                elif arg.endswith("-o"):
                    self.output_free_variables.append(arg[:-2])
                    output_pos_arg[arg[:-2]] = i
                else:
                    assert(arg.endswith("-i"))
                    self.input_free_variables.append(arg[:-2])
                    input_pos_arg[arg[:-2]] = i

        if len(self.output_free_variables) > 0:
            self.rule = defaultdict(list)
            for cval in compliant_values:
                query = tuple([cval[input_pos_arg[x]] for x in self.action_arguments + self.input_free_variables])
                output_query = tuple([cval[output_pos_arg[x]] for x in self.output_free_variables])
                
                self.rule[query].append(output_query)
        else:
            self.rule = set()
            for cval in compliant_values:
                query = tuple([cval[input_pos_arg[x]] for x in self.action_arguments + self.input_free_variables])
                self.rule.add(query)
        # print("XXX", rule, self.action_arguments, self.input_free_variables, self.output_free_variables, self.rule)

    def evaluate(self, action_arguments, free_variable_inputs):
        if free_variable_inputs:
            free_variable_inputs_variables, free_variable_inputs_values = free_variable_inputs
            copy_variables = [var for var in free_variable_inputs_variables if var not in self.input_free_variables]
            copy_values = [free_variable_inputs_variables.index(x) for x in copy_variables]
        else:
            copy_values = []
            
        if not self.input_free_variables:
            query = tuple([action_arguments[x] for x in self.action_arguments])
            if len(self.output_free_variables) == 0:
                result = True if query in self.rule else False
                if result and free_variable_inputs:
                    result = free_variable_inputs
            else:
                if query not in self.rule:
                    result = False
                elif copy_values:
                    print("Not supported", self.rule_text, free_variable_inputs)
                    exit()
                    # result = (output_vars, [values[i] for i in copy_values] + self.rule[query])
                else:
                    return self.output_free_variables, self.rule[query]
            return result
        else:
            output_vars = [var for var in free_variable_inputs_variables if var not in self.input_free_variables] + self.output_free_variables
            output_values = []
            pos_input_free_variables = [free_variable_inputs_variables.index(x) for x in self.input_free_variables]
            for values in free_variable_inputs_values:
                query = tuple([action_arguments[x] for x in self.action_arguments] + [values[x] for x in pos_input_free_variables])
                if len(self.output_free_variables) == 0:
                    if query in self.rule:
                        if copy_values:
                            output_values.append([values[i] for i in copy_values])
                        else:
                            return True
                else:
                    if query in self.rule:
                        output_val = [values[i] for i in copy_values] + self.rule[query] 
                        output_values.append(output_val)
            return output_vars, output_values


def construct_aleph_tree(text, task):
    return AlephRuleConstant(text) if is_float(text) else AlephRuleTree(text, task)


class AlephRuleTree:
    def __init__(self, text, task):
        text_rule, text_false = text.split(";")[0].strip().split(" ")
        text_true = ";".join(text.split(";")[1:])
        
        self.rule = AlephRule (text_rule, task)
        self.case_true = construct_aleph_tree(text_true, task)
        self.case_false = construct_aleph_tree(text_false, task)
        
    def evaluate(self, action_arguments, free_variable_values = {}):
        eval_rule = self.rule.evaluate(action_arguments, free_variable_values)
        if eval_rule:
            if type(eval_rule) != bool:
                return self.case_true.evaluate(action_arguments, eval_rule)
            else:
                return self.case_true.evaluate(action_arguments, None)
        else:
            return self.case_false.evaluate(action_arguments, free_variable_values)
    

class AlephRuleConstant:
    def __init__(self, text):
        self.value = float(text)
        
    def evaluate(self, action_arguments, free_variables_inputs = None):
        return self.value


class RuleEvaluatorAleph:
    def __init__(self, rule_text, task):
        self.rules = {}
        for line in rule_text:
            action_schema, rule = line.split(":-")
            self.rules[action_schema.strip()] = construct_aleph_tree(rule, task)

        # statistics
        self.min_estimate_by_schema = {schema: math.inf for schema in self.rules.keys()}
        self.max_estimate_by_schema = {schema: 0.0 for schema in self.rules.keys()}
        self.sum_estimates_by_schema = {schema: 0.0 for schema in self.rules.keys()}
        self.num_estimates_by_schema = {schema: 0 for schema in self.rules.keys()}
        self.values_off_for_schema = set()

    def get_trained_schemas(self):
        return self.rules.keys()
            
    def get_estimate(self, action):
        schema = action.predicate.name
        estimate = self.rules[schema].evaluate(action.args)
        self.min_estimate_by_schema[schema] = min(estimate, self.min_estimate_by_schema[schema])
        self.max_estimate_by_schema[schema] = max(estimate, self.max_estimate_by_schema[schema])
        self.sum_estimates_by_schema[schema] += estimate
        self.num_estimates_by_schema[schema] += 1
        return estimate

    def get_estimates(self, actions):
        schema = actions[0].predicate.name
        estimates = [self.rules[schema].evaluate(a.args) for a in actions]
        self.min_estimate_by_schema[schema] = min(min(estimates), self.min_estimate_by_schema[schema])
        self.max_estimate_by_schema[schema] = max(max(estimates), self.max_estimate_by_schema[schema])
        self.sum_estimates_by_schema[schema] += sum(estimates)
        self.num_estimates_by_schema[schema] += len(estimates)
        return estimates

    def print_stats(self):
        print("Statistics of Aleph model:")
        print(f"schema{' ' * 9} \t #predictions \t min \t avg \t max")
        for schema in self.rules.keys():
            avg = 0
            if self.num_estimates_by_schema[schema] > 0:
                avg = round(self.sum_estimates_by_schema[schema] / self.num_estimates_by_schema[schema], 2)
            print(f"{schema}{' ' * max(0, 15 - len(schema))} \t "
                  f"{self.num_estimates_by_schema[schema]} \t\t "
                  f"{round(self.min_estimate_by_schema[schema], 2)} \t "
                  f"{avg} \t "
                  f"{round(self.max_estimate_by_schema[schema], 2)}")
