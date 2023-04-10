#! /usr/bin/env python

from __future__ import print_function

import os
import io
import numpy as np
from collections import defaultdict
import shutil
import bz2
import string

from sys import version_info


is_python_3 = version_info[0] > 2 # test python version

try:
    # Python 3.x
    from builtins import open as file_open
except ImportError:
    # Python 2.x
    from codecs import open as file_open

import sys

sys.path.append(f'{os.path.dirname(__file__)}/../translate')
import pddl
from pddl_parser import lisp_parser
from pddl_parser import parsing_functions
import instantiate


def parse_pddl_file(type, filename):
    try:
        # The builtin open function is shadowed by this module's open function.
        # We use the Latin-1 encoding (which allows a superset of ASCII, of the
        # Latin-* encodings and of UTF-8) to allow special characters in
        # comments. In all other parts, we later validate that only ASCII is
        # used.
        return lisp_parser.parse_nested_list(file_open(filename,
                                                       encoding='ISO-8859-1'))
    except IOError as e:
        raise SystemExit("Error: Could not read file: %s\nReason: %s." %
                         (e.filename, e))
    except lisp_parser.ParseError as e:
        raise SystemExit("Error: Could not parse %s file: %s\nReason: %s." %
                         (type, filename, e))



if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("runs_folder", help="path to task pddl file")
    argparser.add_argument("store_training_data", help="Directory to store the training data by gen-subdominization-training")
    argparser.add_argument("--op-file", default="sas_plan", help="File to store the training data by gen-subdominization-training")
    argparser.add_argument("--domain-name", default="domain", help="name of the domain")
    argparser.add_argument("--class-probability", action="store_true", help="write files for class probability, otherwise good/bad files for actions")
    argparser.add_argument("--add-negated-predicates", action="store_true", help="add negation to model")
    argparser.add_argument("--add-equal-predicate", action="store_true", help="add new equal predicate")

    options = argparser.parse_args()

    if os.path.exists(options.store_training_data):
        if (is_python_3):
            result = input('Output path "{}" already exists. Overwrite (Y/n)?'.format(options.store_training_data))
        else:
            result = raw_input('Output path "{}" already exists. Overwrite (Y/n)?'.format(options.store_training_data))
        if result.lower() not in ['y', 'yes', '']:
            sys.exit()
        shutil.rmtree(options.store_training_data)

    operators_filename = options.op_file

    if not os.path.exists(options.store_training_data):
        os.makedirs(options.store_training_data)

    aleph_base_file_content = io.StringIO()

    if (options.class_probability):
        aleph_base_file_content.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
                                      "% specify tree type:\n"
                                      ":- set(tree_type,class_probability).\n"
                                      #":- set(evalfn,entropy). % only alternative when using class_probability is gini\n"
                                      ":- set(classes,[ground,dont_ground]).\n"
                                      #":- set(minpos,2).    % minimum examples in leaf for splitting\n"
                                      #":- set(clauselength,5).\n"
                                      #":- set(lookahead,2).    % to allow lookahead to lteq/2\n"
                                      #":- set(prune_tree,true).\n"
                                      #":- set(confidence,0.25).% pruning conf parameter used by C4.5\n"
                                      ":- set(dependent,DEPENDENT). % second arg of class is to predicted\n\n")

    all_instances = sorted([d for d in os.listdir(options.runs_folder) if os.path.isfile('{}/{}/{}'.format(options.runs_folder, d, operators_filename))])

    # TODO split into training and validation set
#     np.random.seed(2018)
#     testing_instances = np.random.choice(all_instances, int(options.num_test_instances), replace=False)

    domain_filename = '{}/{}/{}'.format(options.runs_folder, all_instances[0], "domain.pddl")
    domain_pddl = parse_pddl_file("domain", domain_filename)
    domain_name, domain_requirements, types, type_dict, constants, predicates, predicate_dict, functions, action_schemas, axioms = parsing_functions.parse_domain_pddl(domain_pddl)
    predicates = [p for p in predicates if p.name != "="]

    aleph_base_file_content.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
                                  "% modes:\n")

    determination_backgrounds = []

    for predicate in predicates:
        arity = len(predicate.arguments)
        for i in range(arity):
            params = "("
            if (arity > 0):
                params += "+'type:" if i == 0 else "-'type:"
                params += predicate.arguments[0].type_name
                params += "'"
                for j in range(1, arity):
                    params += ", +'type:" if j == i else ", -'type:"
                    params += predicate.arguments[j].type_name
                    params += "'"
                params += ", "
            params += "+task_id)"

            aleph_base_file_content.write(":- modeb(*, 'ini:{predicate.name}'{params}).\n".format(**locals()))
            aleph_base_file_content.write(":- modeb(*, 'goal:{predicate.name}'{params}).\n".format(**locals()))
            if (options.add_negated_predicates):
                aleph_base_file_content.write(":- modeb(*, 'ini:not:{predicate.name}'{params}).\n".format(**locals()))
                aleph_base_file_content.write(":- modeb(*, 'goal:not:{predicate.name}'{params}).\n".format(**locals()))
        determination_backgrounds.append("'ini:{name}'/{size}".format(name = predicate.name, size = arity + 1))
        determination_backgrounds.append("'goal:{name}'/{size}".format(name = predicate.name, size = arity + 1))
        if (options.add_negated_predicates):
            determination_backgrounds.append("'ini:not:{name}'/{size}".format(name = predicate.name, size = arity + 1))
            determination_backgrounds.append("'goal:not:{name}'/{size}".format(name = predicate.name, size = arity + 1))

    if (options.add_equal_predicate):
        determination_backgrounds.append("equals/3")
        aleph_base_file_content.write("\n")
        aleph_base_file_content.write(":- modeb(*, equals(+'type:object', +'type:object', +task_id)).\n")

    aleph_base_file_content.write("\n")

    # handle the training instances
    good_operators = defaultdict(lambda : defaultdict(list))
    bad_operators = defaultdict(lambda : defaultdict(list))
    objects = defaultdict(set)

    aleph_fact_file_content = io.StringIO()
    for task_run in all_instances:
        print ("Processing ", task_run)
        domain_filename = '{}/{}/{}'.format(options.runs_folder, task_run, "domain.pddl")
        task_filename = '{}/{}/{}'.format(options.runs_folder, task_run, "problem.pddl")
        plan_filename = '{}/{}/{}'.format(options.runs_folder, task_run, operators_filename)

        domain_pddl = parse_pddl_file("domain", domain_filename)
        task_pddl = parse_pddl_file("task", task_filename)


        task = parsing_functions.parse_task(domain_pddl, task_pddl)

        aleph_fact_file_content.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        aleph_fact_file_content.write("% init {task_run}\n".format(**locals()))
        for ini_fact in task.init:
            if (type(ini_fact) == pddl.Assign or ini_fact.predicate == "="): # we have our own equality
                continue
            aleph_fact_file_content.write("'ini:{ini_fact.predicate}'(".format(**locals()))
            if (len(ini_fact.args) > 0):
                aleph_fact_file_content.write("'obj:{ini_fact.args[0]}'".format(**locals()))
                objects[task_run].add(ini_fact.args[0])
                for arg in ini_fact.args[1:]:
                    objects[task_run].add(arg)
                    aleph_fact_file_content.write(", 'obj:")
                    aleph_fact_file_content.write(arg)
                    aleph_fact_file_content.write("'")
                aleph_fact_file_content.write(", ")
            aleph_fact_file_content.write("'{task_run}').\n".format(**locals()))

        aleph_fact_file_content.write("\n")

        aleph_fact_file_content.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        aleph_fact_file_content.write("% goal " + task_run + "\n")
        for goal_fact in task.goal.parts:
            aleph_fact_file_content.write("'goal:{goal_fact.predicate}'(".format(**locals()))
            if (len(goal_fact.args) > 0):
                aleph_fact_file_content.write("'obj:{goal_fact.args[0]}'".format(**locals()))
                objects[task_run].add(goal_fact.args[0])
                for arg in goal_fact.args[1:]:
                    objects[task_run].add(arg)
                    aleph_fact_file_content.write(", 'obj:")
                    aleph_fact_file_content.write(arg)
                    aleph_fact_file_content.write("'")
                aleph_fact_file_content.write(", ")
            aleph_fact_file_content.write("'" + task_run + "').\n")

        aleph_fact_file_content.write("\n")

        with open(plan_filename, "r") as plan_file:
            plan = set(map(lambda x : tuple(x.replace("\n", "").replace(")", "").replace("(", "").split(" ")), plan_file.readlines()))

            all_operators_filename = '{}/{}/{}'.format(options.runs_folder, task_run, "all_operators")
            all_operators_content = []
            if os.path.isfile(all_operators_filename):
                with open(all_operators_filename, "r") as actions:
                    all_operators_content = [action for action in actions]
            else:
                with bz2.open(all_operators_filename + '.bz2', "rt") as actions:
                    all_operators_content = [action.decode('utf-8') for action in actions]


            # Write good and bad operators
            for action in all_operators_content:
                    schema, arguments = action.split("(")
                    arguments = [x.strip() for x in arguments.strip()[:-1].split(",")]
                    if (tuple([schema] + arguments) in plan): # is in plan
                        good_operators[task_run][schema].append(arguments)
                    else:
                        bad_operators[task_run][schema].append(arguments)


    # write the actual files
    for schema in action_schemas:
        with open(os.path.join(options.store_training_data, options.domain_name + "_" + schema.name + ".b"), "w") as b_file:
            b_file.write(aleph_base_file_content.getvalue().replace("set(dependent,DEPENDENT)", "set(dependent,{D})".format(D=len(schema.parameters) + (2 if options.class_probability else 1))))
            params = "+'type:{schema.parameters[0].type_name}'".format(**locals())
            for param in schema.parameters[1:]:
                params += ", +'type:{param.type_name}'".format(**locals())
            params += ", +task_id"

            if (options.class_probability):
                b_file.write(":- modeh(1, class({params}, -class)).\n\n".format(**locals()))
            else:
                b_file.write(":- modeh(1, class({params})).\n\n".format(**locals()))

            b_file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
            b_file.write("% determinations:\n")
            for bg in determination_backgrounds:
                b_file.write(":- determination(class/{arity}, {pred}).\n".format(arity=len(schema.parameters) + (2 if options.class_probability else 1), pred=bg))

            b_file.write("\n")

            if (options.add_negated_predicates):
                b_file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
                b_file.write("% negated predicates:\n")
                for predicate in predicates:
                    if (len(predicate.arguments) > 26):
                        sys.exit("lazy programmer ERROR")
                    args = [x for x in string.ascii_uppercase[:len(predicate.arguments)]]
                    args_string = args[0]
                    for arg in args[1:]:
                        args_string += ", " + arg
                    b_file.write("'ini:not:{predicate.name}'({args_string}, task):- ".format(**locals()))
                    b_file.write("obj({args[0]}, task)".format(**locals()))
                    for arg in args[1:]:
                        b_file.write(", obj({arg}, task)".format(**locals()))
                    b_file.write(", not('ini:{predicate.name}'({args_string}, task)).\n".format(**locals()))
                    b_file.write("'goal:not:{predicate.name}'({args_string}, task):- ".format(**locals()))
                    b_file.write("obj({args[0]}, task)".format(**locals()))
                    for arg in args[1:]:
                        b_file.write(", obj({arg}, task)".format(**locals()))
                    b_file.write(", not('goal:{predicate.name}'({args_string}, task)).\n".format(**locals()))

                b_file.write("\n")

                for task in objects.keys():
                    b_file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
                    b_file.write("% all objects of task {task}:\n".format(**locals()))
                    for obj in objects[task]:
                        b_file.write("obj('{obj}', '{task}').\n".format(**locals()))
                    b_file.write("\n")

            if (options.add_equal_predicate):
                for task in objects.keys():
                    b_file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
                    b_file.write("% equals task {task}:\n".format(**locals()))
                    for obj in objects[task]:
                        b_file.write("equals('obj:{obj}', 'obj:{obj}', '{task}').\n".format(**locals()))
                    b_file.write("\n")

                b_file.write("\n")

            b_file.write(aleph_fact_file_content.getvalue())


        with open(os.path.join(options.store_training_data, options.domain_name + "_" + schema.name + ".f"), "w") as f_file:
            f_file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
            f_file.write("% training data {s}:\n".format(s=schema.name))

            for task in set(good_operators.keys()).union(set(bad_operators.keys())):
                for arguments in good_operators[task][schema.name]:
                    f_file.write("class('obj:{a}'".format(a=arguments[0]))
                    for arg in arguments[1:]:
                        f_file.write(", 'obj:{arg}'".format(**locals()))
                    if (options.class_probability):
                        f_file.write(", '{task}', ground).\n".format(**locals()))
                    else:
                        f_file.write(", '{task}').\n".format(**locals()))

                if (options.class_probability):
                    for arguments in bad_operators[task][schema.name]:
                        f_file.write("class('obj:{a}'".format(a=arguments[0]))
                        for arg in arguments[1:]:
                            f_file.write(", 'obj:{arg}'".format(**locals()))
                        f_file.write(", '{task}', dont_ground).\n".format(**locals()))

        if (not options.class_probability):
            with open(os.path.join(options.store_training_data, options.domain_name + "_" + schema.name + ".n"), "w") as n_file:
                n_file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
                n_file.write("% training data {s}:\n".format(s=schema.name))
                for task in set(good_operators.keys()).union(set(bad_operators.keys())):
                    for arguments in bad_operators[task][schema.name]:
                        n_file.write("class('obj:{a}'".format(a=arguments[0]))
                        for arg in arguments[1:]:
                            n_file.write(", 'obj:{arg}'".format(**locals()))
                        n_file.write(", '{task}').\n".format(**locals()))


    aleph_base_file_content.close()
    aleph_fact_file_content.close()
