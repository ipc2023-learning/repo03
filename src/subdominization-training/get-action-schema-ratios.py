#! /usr/bin/env python

import argparse
import os
from _collections import defaultdict
from numpy import median
import bz2


def read_actions(file):
    actions = defaultdict(set)
    with bz2.open(file, "rt") if file.endswith(".bz2") else open(file, "r") as f:
        for l in f:
            line = l.strip()
            if (line.startswith(";")):
                # probably cost-info line at end of sas_plan
                return actions
            if (line.startswith("(") and line.endswith(")")):
                # assume format: (move-b-to-t b4 b2)
                schema = line[1:line.index(" ")].strip()
                args = line[line.index(" "):-1].strip()
            elif("(" in line and ")" in line):
                # assume format: move-b-to-t(b4, b2)
                schema = line[:line.index("(")].strip()
                args = line[line.index("(") + 1:line.index(")")].strip()
            else:
                # assume format: move-b-to-t b2 b3
                schema = line[:line.index(" ")].strip()
                args = line[line.index(" "):].strip()
            args = args.replace(", ", " ")
            args = args.replace(",", " ")
            if (args in actions[schema]):
                print(f"WARNING: duplicate action {schema} {args} in file: {file}")
            actions[schema].add(args)
    return actions
            


if __name__ == "__main__":
        
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--runs-folder", help="Path that contains one folder per training instance with the corresponding action files.", required=True)
    argparser.add_argument("--aggregate", help="How to aggregate the ratios over instances? [median, min, max, avg]", default="avg")
    argparser.add_argument("--plan-file-name", help="Name of files containing all actions of a plan of an instance.", default="sas_plan")
    argparser.add_argument("--all-actions-file-name", help="Name of files containing all actions of an instance.", default="all_operators.bz2")
    argparser.add_argument("--output-file", help="Where to write the ratios? Schema with aggregated ratio 0 are not added.")
    argparser.add_argument("--plan-ratios", action="store_true", 
                           help="If set, the ratio is computed on the number of actions in a plan," 
                                "ignoring the total number of actions in the grounding.")    
    
    options = argparser.parse_args()
    
    if (options.aggregate not in ["median", "min", "max", "avg"]):
        sys.exit(f"ERROR: unknown aggregation method: {options.aggregate}; options are median, min, max, avg")
        
    runs_folder = options.runs_folder
    if (not os.path.isdir(runs_folder)):
        sys.exit(f"ERROR: not a folder.. {runs_folder}")
    
    
    ratios = defaultdict(list)
    
    for inst_folder in os.listdir(runs_folder):
        
        plan_file = os.path.join(runs_folder, inst_folder, options.plan_file_name)
        all_actions_file = os.path.join(runs_folder, inst_folder, options.all_actions_file_name)
        
        if (not os.path.isfile(plan_file)):
            print(f"Skipping folder {inst_folder}, does not contain plan file ({options.plan_file_name})")
            continue
        if (not options.plan_ratios and not os.path.isfile(all_actions_file)):
            print(f"Skipping folder {inst_folder}, does not contain all-actions file ({options.all_actions_file})")
            continue
        
        plan_actions = read_actions(plan_file)
        if (options.plan_ratios):
            num_plan_actions = sum([len(plan_actions[schema]) for schema in plan_actions])
        else:
            all_actions = read_actions(all_actions_file)
        
        for schema in plan_actions:
            if (options.plan_ratios):
                ratios[schema].append(len(plan_actions[schema]) / num_plan_actions)
            else:
                if (not plan_actions[schema].issubset(all_actions[schema])):
                    print("WARNING: plan actions are not a subset of all actions")
                ratios[schema].append(len(plan_actions[schema]) / len(all_actions[schema]))
        
        if (not options.plan_ratios):
            for schema in all_actions:
                if (not schema in plan_actions):
                    ratios[schema].append(0.0)
        
    skipped_schemas = []
    outfile = None
    if (options.output_file):
        if (not os.path.isfile(options.output_file) or input(f"Output file {options.output_file} already exists. Overwrite (Y/n)?") in ["", "Y", "y"]):                        
            outfile = open(options.output_file, "w")
    for schema in ratios:
        if (options.aggregate == "avg"):
            agg = sum(ratios[schema]) / len(ratios[schema])
        elif (options.aggregate == "median"):
            agg = median(ratios[schema])
        elif (options.aggregate == "min"):
            agg = min(ratios[schema])
        elif (options.aggregate == "max"):
            agg = max(ratios[schema])
        else:
            sys.exit(f"ERROR: unknown aggregation method: {options.aggregate}; options are median, min, max, avg")
        
        if (agg != 0):
            print(f"{schema}:{agg}")
            if (outfile):
                outfile.write(f"{schema}:{agg}\n")
        else:
            skipped_schemas.append(schema)
    if (outfile):
        outfile.close()
    
    if (skipped_schemas):
        print("No action in plan from the following schemas:")
    for schema in skipped_schemas:
        print(schema)
            
        
        
        
        