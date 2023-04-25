from collections import defaultdict
import itertools

import sys
import os

sys.path.append(f'{os.path.dirname(__file__)}/../../translate')
import pddl

def  valid_values(variables, values, variable_domains):
        assert (len (variables) == len(values))#, "Error: {} {}".format(str(variables), str(values)))
        return all(values[i] in variable_domains[var] for i, var in  enumerate (variables))


class FreeVariableConstraint:
        def __init__(self, action_arguments_rule, free_variables, compliant_values):
                self.action_arguments = action_arguments_rule
                self.free_variables = free_variables
                self.compliant_values = compliant_values

        def action_args_domains (self):
                domains = {}
                for arg in self.action_arguments:
                        domains[arg] = set()

                for values in self.compliant_values:
                        for i, arg in enumerate(self.action_arguments):
                                domains[arg].add (values[i])
                return domains

        def get_free_variable_domains(self):
                domains = {}
                for arg in self.free_variables:
                        domains[arg] = set()

                for values in self.compliant_values:
                        for i, arg in enumerate(self.free_variables):
                                domains[arg].add (values[len(self.action_arguments):][i])

                return domains

        def update(self, free_variable_domains, action_argument_domains):
                new_set = set([x for x in self.compliant_values if valid_values(self.action_arguments, x[:len(self.action_arguments)], action_argument_domains)
                               and valid_values(self.free_variables, x[len(self.action_arguments):], free_variable_domains)])

                # for x in self.compliant_values:
                #         if x not in new_set:
                #                 print "Eliminated", x
                #                 print free_variable_domains
                #                 print action_argument_domains

                if new_set != self.compliant_values:
                        self.compliant_values = new_set
                        return True
                return False

        def evaluate(self, arguments, free_var_values):
                values = tuple(map(lambda x : arguments[x],  self.action_arguments)) + tuple(map(lambda x : free_var_values[x],  self.free_variables))
                return values in self.compliant_values


# class FreeVariableConstraint:
#         def __init__(self, action_arguments_rule, compliant_values, free_variables, arguments):
#                 self.free_variables = free_variables
#                 self.action_arguments = action_arguments_rule

#                 self.rule = {}

#                 pos_free_vars = [arguments.index(free_var) for free_var in free_variables]
#                 for val in compliant_values:
#                         val_free_vars = tuple([val[pos_free_var] for pos_free_var in pos_free_vars])
#                         val_bind_vars = tuple([val[i] for i in range(len(val)) if i not in pos_free_vars])
#                         if val_bind_vars not in self.rule:
#                                 self.rule [val_bind_vars] = set()

#                         self.rule [val_bind_vars].add(val_free_vars)


#         def free_variable_domains(self):
#                 domains_rule = {}
#                 for fv in self.free_variables:
#                         domains_rule[fv] = set()

#                 for _, values in self.rule.items():
#                         for val in values:
#                                 for i, fv in enumerate(self.free_variables):
#                                         domains_rule[fv].add(val[i])

#                 return domains_rule

#         def action_args_domains (self):
#                 domains = {}
#                 for arg in self.action_arguments:
#                         domains[arg] = set()

#                 for values in self.rule:
#                         for i, fv in enumerate(self.action_arguments):
#                                 domains[fv].add(values[i])

#                 return domains

#         def update(self, free_variable_domains, action_argument_domains):
#                 changes = False
#                 invalid_values = [act_args_values for act_args_values in self.rule if not valid_values (self.action_arguments, act_args_values, action_argument_domains)]
#                 for values in invalid_values:
#                         del self.rule[values]
#                         changes = True

#                 for act_args_values in self.rule:
#                         for free_var_values in self.rule[act_args_values]:
#                                 new_set = set([x for x in self.rule[act_args_values] if valid_values(self.free_variables, x, free_variable_domains)])
#                                 if new_set != self.rule[act_args_values]:
#                                         self.rule[act_args_values] = new_set
#                                         changes = True
#                 return changes

#         def evaluate(self, arguments, free_var_values):
#                 values = tuple(map(lambda x : arguments[x],  self.action_arguments))
#                 exit()
#                 return values in self.compliant_values

class Constraint:
        def __init__(self, action_arguments_rule, compliant_values):
                self.action_arguments = action_arguments_rule
                self.compliant_values = compliant_values
                self.free_variables = []

        def action_args_domains (self):
                domains = {}
                for arg in self.action_arguments:
                        domains[arg] = set()

                for values in self.compliant_values:
                        for i, arg in enumerate(self.action_arguments):
                                domains[arg].add (values[i])
                return domains

        def get_free_variable_domains(self):
                return {}

        def update(self, free_variable_domains, action_argument_domains):
                new_set = set([x for x in self.compliant_values if valid_values(self.action_arguments, x, action_argument_domains)])
                if new_set != self.compliant_values:
                        self.compliant_values = new_set
                        return True
                return False

        def evaluate(self, arguments):
                values = tuple(map(lambda x : arguments[x],  self.action_arguments))
                return values in self.compliant_values


def evaluate_inigoal_rule(rule, fact_list):
        def eval_constants(fact, constants):
            for (i, val) in constants:
                if fact.args[i] != val:
                    return False
            return True
        compliant_values = set()

        predicate_name, arguments  = rule.split("(")
        arguments = arguments.replace(")", "").replace("\n", "").replace(".", "").replace(" ", "").split(",")
        if arguments == ['']:
            # e.g. for predicates without argument like "handempty()"
            arguments = []
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

def get_free_variable_domains (constraints):
    free_variable_domains = {}
    for c in constraints:

        for fv in free_vars:
                if fv in free_variable_domains:
                        free_variable_domains[fv] = free_variable_domains[fv].intersection(domains_rule[fv])
                else:
                        free_variable_domains[fv] = domains_rule[fv]

    return free_variable_domains



class RuleEval:
    def __init__(self, rule_text, task):
        #print("Loading: " + rule_text)
        self.text = rule_text.replace('\n','')
        head, body = rule_text.split(":-")
        self.action_schema, action_arguments = head.split(" (")
        self.constraints = []

        action_arguments = action_arguments.replace(")", "").replace("\n", "").replace(".", "").replace(" ", "").split(",")

        for rule in body.split(";"):
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

                # TODO : Support super types in equality rules
                compliant_values = set([tuple ([o.name for a in arguments])
                                        for o in task.objects if o.type_name in argument_types])

            elif rule_type == "true":
                continue
            else:
                 print("Error: unknown rule ", rule_type, rule)
                 exit()

            action_arguments_rule = tuple(map(lambda x : action_arguments.index(x),  filter(lambda x : x in action_arguments, arguments)))
            free_variables = tuple (filter(lambda x : x not in action_arguments, arguments))

            if len(free_variables) == 0:
                self.constraints.append(Constraint(action_arguments_rule, compliant_values))
            else:
                pos = tuple(filter(lambda i : arguments[i] in action_arguments, range(len(arguments)))) + tuple(filter(lambda i : arguments[i] not in action_arguments, range(len(arguments))))
                compliant_values = list(map(lambda x : tuple ([x[i] for i in pos]), compliant_values))
                self.constraints.append(FreeVariableConstraint(action_arguments_rule, free_variables, compliant_values))

        self.set_domain()

        while self.free_variable_domains:
                self.eliminate_free_variable()


    def eliminate_free_variable(self):
            fv, old_domain = self.free_variable_domains.popitem()
            old_constraints = [c for c in self.constraints if fv in c.free_variables]

            new_action_arguments = list(set.union(*[set(c.action_arguments) for c in old_constraints]))
            new_free_variables = set.union(*[set(c.free_variables) for c in old_constraints])
            new_free_variables.remove(fv)
            new_free_variables = list(new_free_variables)

            pos_fv = [len(c.action_arguments) + c.free_variables.index(fv) for c in old_constraints]

            new_compliant_values = set()

            for val in old_domain:
                    constraints_tuples  = [set([tup for tup in c.compliant_values if tup[pos_fv[i]] == val]) for i, c in enumerate(old_constraints)]
                    for combination in itertools.product(*constraints_tuples):
                            new_tuple = {}
                            conflict = False
                            for i, c in enumerate(old_constraints):
                                    for j, arg in enumerate(c.action_arguments):
                                            if arg in new_tuple and new_tuple[arg] != combination[i][j]:
                                                    conflict = True
                                                    break
                                            new_tuple[arg] = combination[i][j]
                                    if conflict:
                                             break
                                    for j, arg in enumerate(c.free_variables):
                                            if arg == fv:
                                                    continue
                                            if arg in new_tuple and new_tuple[arg] != combination[i][len(c.action_arguments) + j]:
                                                    conflict = True
                                                    break
                                            new_tuple[arg] = combination[i][len(c.action_arguments) + j]
                                    if conflict:
                                            break


                            if conflict:
                                    continue

                            new_tuple = tuple([new_tuple[arg] for arg in new_action_arguments] + [new_tuple[arg] for arg in new_free_variables])
                            new_compliant_values.add(new_tuple)


            if new_free_variables:
                    new_constraint = FreeVariableConstraint(new_action_arguments, new_free_variables, new_compliant_values)
            else:
                    new_constraint = Constraint(new_action_arguments, new_compliant_values)

            self.constraints = [c for c in self.constraints if fv not in c.free_variables] + [new_constraint]

    def set_domain(self):
        self.action_argument_domains = {}
        self.free_variable_domains = {}

        self.update_domain()

        changes = True
        while changes:
                changes = False
                for r in self.constraints:
                        if r.update(self.free_variable_domains, self.action_argument_domains):
                                changes = True
                if changes:
                        self.update_domain()

    def update_domain(self):
            for rule in self.constraints:
                for (fv, values) in rule.get_free_variable_domains().items():
                        if fv not in self.free_variable_domains:
                                self.free_variable_domains [fv] = values
                        else:
                                self.free_variable_domains [fv] = self.free_variable_domains [fv].intersection(values)

                for (arg, values) in rule.action_args_domains().items():
                        if arg not in self.action_argument_domains:
                                self.action_argument_domains [arg] = values
                        else:
                                self.action_argument_domains [arg] = self.action_argument_domains [arg].intersection(values)


        #print (self.text, self.constraints)

    def evaluate(self, arguments):
        if self.free_variable_domains:
                #for fv_values in itertools.product(*[valueset for x, valueset in self.free_variable_domains.items()]):
                return 0
        else:
                for c in self.constraints:
                        if not c.evaluate(arguments):
                                return 0

                #print (action.name, "valid according to", self.text)
                #print ("Evaluate", self.text, action, 1)
                return 1

class RulesEvaluator:
    def __init__(self, rule_text, task):
        self.rules = defaultdict(list)
        for l in rule_text:
            re = RuleEval(l, task)
            self.rules[re.action_schema].append(re)

    def eliminate_rules(self, rules_text):
        for a in self.rules:
            self.rules[a] = [rule for rule in self.rules[a] if rule.text not in rules_text]

    def evaluate(self, action_name, arguments):
        return [rule.evaluate(arguments) for rule in  self.rules[action_name]]

    def get_all_rules (self):
        return [rule.text for (schema, rules)  in self.rules.items() for rule in rules]

    def get_action_schemas_with_rules(self):
        return self.rules.keys()
