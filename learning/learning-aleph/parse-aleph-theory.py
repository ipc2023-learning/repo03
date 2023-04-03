#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("directory", help="Domain file")

options = argparser.parse_args()

for schema in os.listdir(options.directory):
    f = open ("{}/{}".format(options.directory, schema))
    theory = False
    lines = f.readlines()
    lines = lines[lines.index("[theory]\n") + 1:]
    lines = lines[:lines.index("[Training set performance]\n")]

    rules = []
    previous_rule = None
    class_args = None
    for l in lines:
        if l.startswith("[Rule"):
            if previous_rule:
                rules.append(previous_rule)
            previous_rule = ""
        elif l.strip():
            previous_rule += l.strip()
        if not class_args and l.startswith("class"):
            class_args = l.split(":-")[0][6:].split(",")[:-2]
    rules.append(previous_rule)


    rules_tuples = []
    for r in rules[::-1]:
        task_arg = r.split(":-")[0].split(",")[-2]
        #        print ("Rule", r)
        if "not" in r:                
            r = "".join(r.replace(":-not", ":-") [r.index(":-") + 2:].replace(",not", ", not").split(", not") [-1:]).replace(", random", ",random")
            predicate = r.split(",random")[0].replace("'", "").replace("," + task_arg, "").strip()
            ground_prob = [x for x in r.split(",") if "-ground" in x][0]
            #print ("Prob: ", ground_prob)
            ground_prob = float(ground_prob[:ground_prob.index("-ground")].replace("[", ""))
            rules_tuples.append((predicate, ground_prob))
        else:
            ground_prob = [x for x in r.split(",") if "-ground" in x][0]
            ground_prob = float(ground_prob[:ground_prob.index("-ground")].replace("[", ""))
            rules_tuples.append(("", ground_prob))


            #print(schema + "(" +  ",".join(class_args) +   ")  " + "; ".join(["{} {:f}".format(x[0], x[1]) for x in rules_tuples]))

    free_vars_args = {}
    num_free_vars_args = 0
    for (i, (r, g)) in enumerate(rules_tuples):
        predicates = r.split("),")
        for pred in predicates:
            if pred.startswith("("):
                pred = pred[1:]
            if not pred:
                continue
            pred_name, args = pred.replace(")", "").split("(")
            for arg in args.split(","):
                if arg in class_args:
                    continue
                if not arg in free_vars_args:
                    num_free_vars_args += 1
                    id_arg = num_free_vars_args
                    first_time = i
                else:
                    (id_arg, first_time, _) = free_vars_args[arg]
                    
                free_vars_args[arg] = (id_arg, first_time, i)

    new_rule_tuples = []
    for (i, (r, g)) in enumerate(rules_tuples):
        predicates = r.split("),")
        new_predicates = []
        for pred in predicates:
            if pred.startswith("("):
                pred = pred[1:]
            if not pred:
                continue
            pred_name, args = pred.replace(")", "").split("(")

            new_args = []
            for arg in args.split(","):
                if arg in class_args:
                    new_args.append("?arg{}".format(class_args.index(arg)))
                else:
                    (id_arg, first_time, last_time) = free_vars_args[arg]
                    if first_time == last_time:
                        name_arg = "_"
                    elif i == last_time:
                        name_arg = "?fv{}-i".format(id_arg)
                    elif i == first_time:
                        name_arg = "?fv{}-o".format(id_arg)
                    else:
                        name_arg = "?fv{}-io".format(id_arg)

                    new_args.append(name_arg)

            new_predicates.append("{}({})".format(pred_name, ",".join(new_args)))

        new_r = ",".join(new_predicates)
        new_rule_tuples.append((new_r, g))
                    
    print(schema + " :- " + "; ".join(["{} {:f}".format(x[0], x[1]) for x in new_rule_tuples]))
    

    
    #print (schema, ";".join(map(lambda x : str(x), rules_tuples)))
    # print (schema, ";".join(map(lambda x : , rules_tuples)))

