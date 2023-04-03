#! /usr/bin/env python

from __future__ import print_function

import os
import io
import numpy as np
from pddl_parser import parsing_functions

from collections import defaultdict
from rule_training_evaluator import *
import lisp_parser
import shutil
import bz2
import string

import pickle

import csv

from sys import version_info


is_python_3 = version_info[0] > 2 # test python version

try:
    # Python 3.x
    from builtins import open as file_open
except ImportError:
    # Python 2.x
    from codecs import open as file_open



if __name__ == "__main__":
    import argparse
    
    argparser = argparse.ArgumentParser()   
    argparser.add_argument("--runs-folder", required=True, help="folder containing the data to evaluate")
    argparser.add_argument("--model-folder", required=True, help="folder containing the models used for evaluation")

    options = argparser.parse_args()

    if (not os.path.exists(options.runs_folder)):
        sys.exit("path does not exist: ", options.runs_folder)
    if (not os.path.exists(options.model_folder)):
        sys.exit("file does not exist: ", options.model_folder)

    models = {}

    for f in os.listdir(options.model_folder):
        if (f.endswith(".model")):
            with open(os.path.join(options.model_folder, f), "rb") as modelFile:
                models[f[:-6]] = pickle.load(modelFile)

    if (len(models) == 0):
        sys.exit("no trained models in ", options.model_folder)

    
    for f in os.listdir(options.runs_folder):
        if (f.endswith(".csv")):
            eval_data_file = open(os.path.join(options.runs_folder, f))
            schema = f[:-4]                
        elif(f.endswith(".csv.bz")):
            eval_data_file = bz2.BZ2File(os.path.join(options.runs_folder, f), "r")
            schema = f[:-7]
        else:
            continue
        
        if (not schema in models):
            print("skipping schema", schema)
            continue
        
        eval_data = csv.reader(eval_data_file, delimiter=",")
        
        error = 0.0
        n = 0
            
        for sample in eval_data:
            if (models[schema].is_classifier):
                error += pow(models[schema].model.predict_proba([list(map(int, sample[:-1]))])[0][1] - int(sample[-1]), 2)
            else:
                error += pow(models[schema].model.predict([list(map(int, sample[:-1]))])[0] - int(sample[-1]), 2)
            n += 1
        
        error = error / n
        
        print("MSE of ", schema, ": ", error)
                
                
                
                
                
                
                
                
                


