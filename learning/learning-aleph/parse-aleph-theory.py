#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse

import utils

argparser = argparse.ArgumentParser()
argparser.add_argument("directory", help="Domain file")
argparser.add_argument("--class-probability", action="store_true", help="this is the result of class probability rules")

options = argparser.parse_args()


def parse_aleph_hypothesis_file(hypothesis_file, is_class_probability):
    with open (hypothesis_file) as f:
        lines = f.readlines()

        rules = []
        class_args = None
        for l in lines:
            if l.startswith("class"):
                class_args = l.split(":-")[0][6:].split(",")[:-2 if is_class_probability else -1]
                rules.append(l.strip())
            else:
                rules[-1] += l.strip()

        return rules, class_args

def parse_aleph_log_file(logfile):
    with open (logfile) as f:
        theory = False
        lines = f.readlines()
        lines = lines[lines.index("[theory]\n") + 1:]
        lines = lines[:lines.index("[Training set performance]\n")]

        rules = []
        previous_rule = None
        class_args = None
        for l in lines:
            if not class_args and l.startswith("class"):
                class_args = l.split(":-")[0][6:].split(",")[:-2 if is_class_probability else -1]

            if l.startswith("[Rule"):
                if previous_rule:
                    rules.append(previous_rule)
                previous_rule = ""
            elif l.strip():
                previous_rule += l.strip()

        if previous_rule:
            rules.append(previous_rule)

    return rules, class_args



# Get those schemas for which we have some hypothesis file
schemas = [fname[:-2] for fname in os.listdir(options.directory) if fname.endswith('.h')]
for schema in schemas:
    rules, class_args = parse_aleph_hypothesis_file("{}/{}.h".format(options.directory, schema), options.class_probability)

    if options.class_probability:
        new_rule_tuples = utils.transform_probability_class_rules(rules, class_args)
        print(schema + " :- " + "; ".join(["{} {:f}".format(x[0], x[1]) for x in new_rule_tuples]))
    else:
        for r in rules:
            print(schema + " (" + ", ".join(["?arg{}".format(i) for i in range(len(class_args))]) + ")" + " :- " + utils.transform_hard_rule (r, class_args) + ".")



    #print (schema, ";".join(map(lambda x : str(x), rules_tuples)))
    # print (schema, ";".join(map(lambda x : , rules_tuples)))
