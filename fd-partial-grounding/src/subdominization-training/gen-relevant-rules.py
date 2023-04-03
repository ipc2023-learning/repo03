#! /usr/bin/env python

from __future__ import print_function

from collections import defaultdict
from rule_training_evaluator import *
import lisp_parser
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
    argparser.add_argument("output", type=argparse.FileType('w'), help="Output file")    
    argparser.add_argument("--instances-relevant-rules", type=int, help="Number of instances for relevant rules", default=1000)    
    argparser.add_argument("--max-training-examples", type=int, help="Maximum number of training examples for action schema", default=1000000)    
    options = argparser.parse_args()

    training_lines = defaultdict(list)
    relevant_rules = set()


    training_re = RuleTrainingEvaluator(options.training_rules.readlines())
    i = 1
    for task_run in sorted(os.listdir(options.runs_folder)):
        if i > options.instances_relevant_rules:
             break
        if not os.path.isfile('{}/{}/{}'.format(options.runs_folder, task_run, 'sas_plan')):
            continue

        print (i)
        i += 1
        
        
        domain_filename = '{}/{}/{}'.format(options.runs_folder, task_run, "domain.pddl")
        task_filename = '{}/{}/{}'.format(options.runs_folder, task_run, "problem.pddl")        
        domain_pddl = parse_pddl_file("domain", domain_filename)
        task_pddl = parse_pddl_file("task", task_filename)
        task = parsing_functions.parse_task(domain_pddl, task_pddl)
        
        training_re.init_task(task, options.max_training_examples)

        operators_filename = '{}/{}/{}'.format(options.runs_folder, task_run, "all_operators.bz2")

        with bz2.BZ2File(operators_filename, "r") as actions:
            # relaxed_reachable, atoms, actions, axioms, _ = instantiate.explore(task)
            for action in actions:
                training_re.evaluate(action.strip())                

    training_re.print_statistics()  

    relevant_rules = training_re.get_relevant_rules()
    #print ("Relevant rules: ", len(relevant_rules))

    options.output.write("\n".join(map(lambda x : x.replace('\n', ''), relevant_rules)))

