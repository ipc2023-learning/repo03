#! /usr/bin/env python

from __future__ import print_function

import os
import numpy as np

from collections import defaultdict
from rule_training_evaluator import *
import lisp_parser
import shutil
import bz2

try:
    # Python 3.x
    from builtins import open as file_open
except ImportError:
    # Python 2.x
    from codecs import open as file_open

import parsing_functions
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
    import os
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("runs_folder", help="path to task pddl file")
    argparser.add_argument("training_rules", type=argparse.FileType('r'), help="File that contains the rules used to generate training data by gen-subdominization-training")
    argparser.add_argument("store_training_data", help="Directory to store the training data by gen-subdominization-training")    
    argparser.add_argument("--debug-info", help="Include action name in the file", action="store_true")    
    argparser.add_argument("--instances-relevant-rules", type=int, help="Number of instances for relevant rules", default=0)    
    argparser.add_argument("--op-file", default="sas_plan", help="File to store the training data by gen-subdominization-training")    
    argparser.add_argument("--num-test-instances", type=int,default=0, help="Number of instances reserved for the testing set")
    argparser.add_argument("--max-training-examples", type=int, help="Maximum number of training examples for action schema", default=1000000)    


    options = argparser.parse_args()

    if os.path.exists(options.store_training_data):
        result = raw_input('Output path "{}" already exists. Overwrite (y/n)?'.format(options.store_training_data))
        if result.lower() not in ['y', 'yes']:
            exit()
        shutil.rmtree(options.store_training_data)

        
    relevant_rules = []

    operators_filename = options.op_file


    if options.instances_relevant_rules:
        training_re = RuleTrainingEvaluator(options.training_rules.readlines())
        i = 1
        for task_run in sorted(os.listdir(options.runs_folder)) [::-1]:
            if i > options.instances_relevant_rules:
                break
            if not os.path.isfile('{}/{}/{}'.format(options.runs_folder, task_run, operators_filename)):
                continue
            print (i)
            i += 1

            domain_filename = '{}/{}/{}'.format(options.runs_folder, task_run, "domain.pddl")
            task_filename = '{}/{}/{}'.format(options.runs_folder, task_run, "problem.pddl")

            domain_pddl = parse_pddl_file("domain", domain_filename)
            task_pddl = parse_pddl_file("task", task_filename)

            task = parsing_functions.parse_task(domain_pddl, task_pddl)
            training_re.init_task(task, options.max_training_examples)        
            # relaxed_reachable, atoms, actions, axioms, _ = instantiate.explore(task)
            
            all_operators_filename = '{}/{}/{}'.format(options.runs_folder, task_run, "all_operators.bz2")

            with bz2.BZ2File(all_operators_filename, "r") as actions:
                # relaxed_reachable, atoms, actions, axioms, _ = instantiate.explore(task)
                for action in actions:
                    training_re.evaluate(action.strip())                

            # for action in actions:
            #     training_re.evaluate(action)                

        training_re.print_statistics()  

        relevant_rules = sorted(training_re.get_relevant_rules())

      
        print ("Relevant rules: ", len(relevant_rules))
    else:
        relevant_rules = sorted([l for l in options.training_rules.readlines()])

    if not os.path.exists(options.store_training_data):
        os.makedirs(options.store_training_data)

    output_file = open('{}/relevant_rules'.format(options.store_training_data), 'w')
    output_file.write("\n".join(map(lambda x : x.replace('\n', ''), relevant_rules)))
    output_file.close()


    training_lines = defaultdict(list)
    testing_lines = defaultdict(list)


    all_instances = sorted([d for d in os.listdir(options.runs_folder) if os.path.isfile('{}/{}/{}'.format(options.runs_folder, d, operators_filename))])
    np.random.seed(2018)
    testing_instances = np.random.choice(all_instances, int(options.num_test_instances), replace=False)

    for task_run in all_instances:        
        print ("Processing ", task_run)
        is_test_instance = task_run in testing_instances
        domain_filename = '{}/{}/{}'.format(options.runs_folder, task_run, "domain.pddl")
        task_filename = '{}/{}/{}'.format(options.runs_folder, task_run, "problem.pddl")
        plan_filename = '{}/{}/{}'.format(options.runs_folder, task_run, operators_filename)

        domain_pddl = parse_pddl_file("domain", domain_filename)
        task_pddl = parse_pddl_file("task", task_filename)

        all_operators_filename = '{}/{}/{}'.format(options.runs_folder, task_run, "all_operators.bz2")
           
        task = parsing_functions.parse_task(domain_pddl, task_pddl)
    
        re = RulesEvaluator(relevant_rules, task)

        #relaxed_reachable, atoms, actions, axioms, _ = instantiate.explore(task)
        
        with open(plan_filename) as plan_file:
            plan = set(map (lambda x : tuple(x.replace("\n", "").replace(")", "").replace("(", "").split(" ")), plan_file.readlines()))
            skip_schemas_training = [schema for schema, examples in training_lines.items() if len(examples) >= options.max_training_examples]
            skip_schemas_testing = [schema for schema, examples in testing_lines.items() if len(examples) >= options.max_training_examples]
            with bz2.BZ2File(all_operators_filename, "r") as actions:
            # relaxed_reachable, atoms, actions, axioms, _ = instantiate.explore(task)
                for action in actions:
                    schema, arguments = action.split("(")
                    if not is_test_instance and schema in skip_schemas_training:
                        continue
                    if is_test_instance and schema in skip_schemas_testing:
                        continue
                    
                    
                    arguments = map(lambda x: x.strip(), arguments.strip()[:-1].split(","))
                   
                    is_in_plan = 1 if  tuple([schema] + arguments) in plan else 0
                   
                    eval = re.evaluate(schema, arguments)
                    #print( ",".join(map (str, [action.name] + eval + [is_in_plan])) )
                
                    new_line = ",".join(map (str, eval + [is_in_plan]))
                    if options.debug_info:
                        new_line = ",".join([task_run, action, new_line])

                    if is_test_instance:
                        testing_lines [schema].append(new_line)
                    else:
                        training_lines [schema].append(new_line)
    
    if testing_lines:
        os.makedirs('{}/training'.format(options.store_training_data))

        shutil.move ('{}/relevant_rules'.format(options.store_training_data), '{}/training/relevant_rules'.format(options.store_training_data))
        
        for schema in sorted(training_lines, key=lambda x : (x.count(";"), x)):
            output_file = open('{}/training/{}.csv'.format(options.store_training_data, schema), 'w')
            for line in training_lines[schema]:
                output_file.write(line + "\n")
            output_file.close()

        os.makedirs('{}/testing'.format(options.store_training_data))
        for schema in sorted(testing_lines, key=lambda x : (x.count(";"), x)):
            output_file = open('{}/testing/{}.csv'.format(options.store_training_data, schema), 'w')
            for line in testing_lines[schema]:
                output_file.write(line + "\n")
            output_file.close()
    else:
        for schema in sorted(training_lines, key=lambda x : (x.count(";"), x)):
            output_file = open('{}/{}.csv'.format(options.store_training_data, schema), 'w')
            for line in training_lines[schema]:
                output_file.write(line + "\n")
            output_file.close()


                           
    #print ("Only 0/1 rules: ", len(re.get_only_0_rules()), len(re.get_only_1_rules()), len(re.get_all_rules()))
