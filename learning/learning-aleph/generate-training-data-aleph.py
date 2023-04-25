#! /usr/bin/env python3

import os
import numpy as np
import shutil
import bz2
import string

from sys import version_info

from aleph_background import PredictionType
from aleph_yap_script import  write_yap_file

from builtins import open as file_open

import sys

sys.path.append(f'{os.path.dirname(__file__)}/../translate')
from pddl_parser import lisp_parser
from pddl_parser import parsing_functions
import instantiate

from aleph_background import *

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


def write_examples_file(filename, examples):
    with open(filename, "w") as n_file:
        n_file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        n_file.write("% training data:\n")
        for task in examples:
            for arguments in examples[task]:
                n_file.write(get_class_example(arguments, task))


def write_class_probability_examples_file(filename, good_examples, bad_examples, use_class_probability):
    with open(filename, "w") as f_file:
        f_file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        f_file.write("% training data")

        for task in set(good_examples.keys()).union(set(bad_examples.keys())):
            for arguments in good_examples[task]:
                f_file.write("class('obj:{a}'".format(a=arguments[0]))
                for arg in arguments[1:]:
                    f_file.write(", 'obj:{arg}'".format(**locals()))
                if (use_class_probability):
                    f_file.write(", '{task}', ground).\n".format(**locals()))
                else:
                    f_file.write(", '{task}').\n".format(**locals()))

            for arguments in bad_examples[task]:
                f_file.write("class('obj:{a}'".format(a=arguments[0]))
                for arg in arguments[1:]:
                    f_file.write(", 'obj:{arg}'".format(**locals()))
                f_file.write(", '{task}', dont_ground).\n".format(**locals()))


def get_arg_list(action):
    return action.split("(")[1].replace(")", "").split(",")

def get_class_example(arguments, task):

    result = "class('obj:{a}'".format(a=arguments[0])
    for arg in arguments[1:]:
        result += ", 'obj:{arg}'".format(**locals())
    result += ", '{task}').\n".format(**locals())
    return result



def generate_training_data_aleph(RUNS_DIR, store_training_data, background_file_opts, positive_examples_filename='good_operators', all_ops_file='all_operators',
                                 aleph_directory=f'{os.path.dirname(__file__)}/../aleph',
                                 yap_command = 'yap', extra_parameters = {}, min_positive_instances=1, min_negative_instances=0):

    if not os.path.exists(store_training_data):
        os.makedirs(store_training_data)

        assert os.path.isfile(f"{aleph_directory}/aleph.pl"), "Error: aleph not found at {}/aleph.pl".format(aleph_directory)
        shutil.copy (f"{aleph_directory}/aleph.pl", store_training_data)

    all_instances = sorted([d for d in os.listdir(RUNS_DIR) if os.path.isfile(os.path.join(RUNS_DIR, d, positive_examples_filename))])
    # print (all_instances)
    domain_filename = os.path.join(RUNS_DIR, all_instances[0], "domain.pddl")
    domain_pddl = parse_pddl_file("domain", domain_filename)
    domain_name, domain_requirements, types, type_dict, constants, predicates, predicate_dict, functions, action_schemas, axioms = parsing_functions.parse_domain_pddl(domain_pddl)

    bg_file = BackgroundFile(predicates, type_dict, background_file_opts)

    # handle the training instances
    good_operators = defaultdict(lambda : defaultdict(list))
    bad_operators = defaultdict(lambda : defaultdict(list))

    for task_run in all_instances:
        # print ("Processing ", task_run)
        domain_filename = os.path.join(RUNS_DIR, task_run, "domain.pddl")
        task_filename = os.path.join(RUNS_DIR, task_run, "problem.pddl")
        plan_filename = os.path.join(RUNS_DIR, task_run, positive_examples_filename)

        domain_pddl = parse_pddl_file("domain", domain_filename)
        task_pddl = parse_pddl_file("task", task_filename)


        task = parsing_functions.parse_task(domain_pddl, task_pddl)

        bg_file.read_instance(task_run, task)


        with open(plan_filename, "r") as plan_file:
            plan = set(map(lambda x : tuple(x.replace("\n", "").replace(")", "").replace("(", "").split(" ")), plan_file.readlines()))
            # Write good and bad operators


            all_positive_examples_filename = os.path.join (RUNS_DIR, task_run, all_ops_file)
            all_operators_content = []
            if os.path.isfile(all_positive_examples_filename):
                with open(all_positive_examples_filename, "r") as actions:
                    all_operators_content = [action for action in actions]
            else:
                with bz2.open(all_positive_examples_filename + '.bz2', "rt") as actions:
                    all_operators_content = [action.decode('utf-8') for action in actions]

            for action in all_operators_content:
                    if "(" in action:
                        schema, arguments = action.split("(")
                        arguments = [x.strip() for x in arguments.strip()[:-1].split(",")]
                    else:
                        tmp = [x.strip() for x in action.split(" ")]
                        schema = tmp[0]
                        arguments = tmp[1:]
                    if (tuple([schema] + arguments) in plan): # is in plan
                        good_operators[schema][task_run].append(arguments)
                    else:
                        bad_operators[schema][task_run].append(arguments)


    if background_file_opts.prediction_type == PredictionType.bad_actions:
        positive_examples, negative_examples = bad_operators, good_operators
    else:
        positive_examples, negative_examples = good_operators, bad_operators


    # write the actual files
    for schema in action_schemas:
        num_pos_examples = sum([len(x) for x in positive_examples[schema.name].values()])
        num_neg_examples = sum([len(x) for x in negative_examples[schema.name].values()])

        if num_pos_examples < min_positive_instances or num_neg_examples < min_negative_instances:
            print(f"Skipping {schema.name} due to lack of training data: {num_pos_examples} positive examples and {num_neg_examples} negative examples" )
            continue

        filename = schema.name
        filename_path = os.path.join(store_training_data, filename)

        write_yap_file (os.path.join(store_training_data, "learn-" + filename), yap_command, filename, filename + ".h", background_file_opts.prediction_type, extra_parameters)

        bg_file.write(filename_path + ".b", [param.type_name for param in schema.parameters])

        if (not background_file_opts.use_class_probability()):
            write_examples_file(filename_path + ".f", positive_examples[schema.name])
            write_examples_file(filename_path + ".n", negative_examples[schema.name])
        else:
            write_class_probability_examples_file(filename_path + ".f", positive_examples[schema.name], negative_examples[schema.name], background_file_opts.use_class_probability())


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("runs_folder", help="path to task pddl file")
    argparser.add_argument("store_training_data", help="Directory to store the training data by gen-subdominization-training")
    argparser.add_argument("--op-file", default="good_operators", help="Filename to determine whether to learn from plans (sas_plan) or the set of optimal_operators")
    argparser.add_argument("--all-ops-file", default="all_operators", help="path for finding all operators")

    argparser.add_argument("--prediction-type", type=PredictionType, default='good-actions', help="Decides whether to learn good-action rules, bad-action rules, or a class probability tree")

    # Different configuration settings
    argparser.add_argument("--add-negated-predicates", action="store_true", help="add negation to model")
    argparser.add_argument("--add-equal-predicate", action="store_true", help="add new equal predicate")
    argparser.add_argument("--use-object-types", action="store_true", help="passes information regarding object types to Aleph")
    argparser.add_argument("--determination-type", type=DeterminationType, choices=list(DeterminationType),default='all-out-except-one',
                           help="Decides how to set the determination. All-out-except-one forces that all parts of the rule are related to some argument of the action and/or some previously declared free-variable")

    argparser.add_argument("--aleph-directory", help="Directory where aleph can be found")
    argparser.add_argument("--yap", default='yap', help="Command to execute yap")
    argparser.add_argument("--min-positive-instances", default=1, help="Skip learning if we do not have at least X examples of the positive and the negative class")
    argparser.add_argument("--min-negative-instances", default=0, help="Skip learning if we do not have at least X examples of the positive and the negative class")



    options = argparser.parse_args()

    if os.path.exists(options.store_training_data):
    #     result = input('Output path "{}" already exists. Overwrite (Y/n)?'.format(options.store_training_data))

    #     if result.lower() not in ['y', 'yes', '']:
    #         sys.exit()

        shutil.rmtree(options.store_training_data)


    bg_opts = BackgroundFileOptions (options.add_negated_predicates, options.add_equal_predicate, options.use_object_types, options.prediction_type, options.determination_type)

    generate_training_data_aleph(options.runs_folder, options.store_training_data, bg_opts, positive_examples_filename=options.op_file, all_ops_file=options.all_ops_file, aleph_directory=options.aleph_directory if options.aleph_directory else f'{os.path.dirname(__file__)}/../aleph', yap_command=options.yap, min_positive_instances=options.min_positive_instances, min_negative_instances=options.min_negative_instances)
