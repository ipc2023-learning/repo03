#!/usr/bin/python


import json
import argparse
import os
import shutil

parser = argparse.ArgumentParser(description='Renames dirs in lab folder')
parser.add_argument('--lab', help='lab experiments dir', required=True)
parser.add_argument('--out', help='output dir', required=True)
parser.add_argument('--planner', help='only results of planner')
parser.add_argument('--include-domain-name', help='', action='store_true')

options = parser.parse_args()

lab_dir = options.lab

assert (os.path.isdir(lab_dir))


for direc in os.listdir(lab_dir):
    if direc.startswith("runs") and os.path.isdir("%s/%s" % (lab_dir, direc)):
        for rundir in os.listdir("%s/%s" % (lab_dir, direc)):
            input_dir = '%s/%s/%s' % (lab_dir, direc, rundir)
            alg, domain, task = json.load(open('%s/properties' % input_dir))['id']

            if options.planner and alg != options.planner:
                continue
            task_name = task.replace('.pddl','')
            
            if options.include_domain_name:
                output_dir = '%s/%s-%s' % (options.out, domain, task_name)
            else:
                output_dir = '%s/%s' % (options.out, task_name)

            shutil.copytree(input_dir, output_dir)
            # print (rundir)

            
