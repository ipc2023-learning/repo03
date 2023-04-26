#! /usr/bin/env python

from collections import defaultdict
import bz2
import sys
import os

sys.path.append(f'{os.path.dirname(__file__)}')
from rule_evaluator.rule_evaluator import *

sys.path.append(f'{os.path.dirname(__file__)}/../translate')
import pddl_parser.lisp_parser as lisp_parser
import pddl_parser.parsing_functions as parsing_functions

from builtins import open as file_open

def parse_pddl_file(type, filename):
    try:
        # The builtin open function is shadowed by this module's open function.
        # We use the Latin-1 encoding (which allows a superset of ASCII, of the
        # Latin-* encodings and of UTF-8) to allow special characters in
        # comments. In all other parts, we later validate that only ASCII is
        # used.
        return lisp_parser.parse_nested_list(filename)
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
    argparser.add_argument("domain", type=argparse.FileType('r'), help="path to domain file")
    argparser.add_argument("problem", type=argparse.FileType('r'), help="path to problem file")
    argparser.add_argument("operators", type=argparse.FileType('r'), help="path to list of actions that should be filtered")

    argparser.add_argument("rules", type=argparse.FileType('r'), help="File that contains the rules used to generate training data by gen-subdominization-training")
    argparser.add_argument("output", type=argparse.FileType('w'), help="Output file")

    options = argparser.parse_args()


    domain_pddl = parse_pddl_file("domain", options.domain)
    task_pddl = parse_pddl_file("task", options.problem)
    task = parsing_functions.parse_task(domain_pddl, task_pddl)
    reval = RulesEvaluator(options.rules.readlines(), task)
    for action in options.operators.readlines():
        #print(action)
        if "(" in action:
            name, arguments = action.split("(")
            arguments = list(map(lambda x: x.strip(), arguments.strip()[:-1].split(",")))
        else:
            args = action.strip().split(" ")
            name = args[0]
            arguments = list(map(lambda x: x.strip(), args[1:]))
        #print(name, arguments)
        if not any(reval.evaluate(name, arguments)):
            options.output.write(action)
