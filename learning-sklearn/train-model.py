#!/usr/bin/python

from subdominization.learn import LearnRules

import os
import sys
import argparse

from shutil import copy

from sys import version_info



def train_model(model_folder, training_folder, model_type, keep_duplicate_features, mean_over_duplicates, yes_to_all=False):
    is_python_3 = version_info[0] > 2 # test python version
    if (not os.path.isdir(model_folder)): # create output folder if does not exist
        if (not yes_to_all):
            if (is_python_3):
                response = input("create output folder \"%s\"? Y/n" % model_folder)
            else:
                response = raw_input("create output folder \"%s\"? Y/n" % model_folder)
        if (yes_to_all or not response in ["no", "No", "NO", "n", "N"]):
            os.makedirs(model_folder)
        else:
            print("ERROR: output folder does not exist")
            sys.exit(1)
    
    # copy relevant_rules files to model_folder
    relevant_rules_file = os.path.join(training_folder, "relevant_rules")
    if (os.path.isfile(relevant_rules_file)):
        copy(relevant_rules_file, model_folder)
    else:
        print("WARNING: no \"relevant_rules\" file in training folder")
    
    overwrite_existing_files = None
    
    # generate models and save to file
    for file in os.listdir(training_folder):
        curr_file = os.path.join(training_folder, file)
        if (os.path.isfile(curr_file) and (file.endswith(".csv") or file.endswith(".csv.bz2"))):
            name = file[:-4] + ".model" if file.endswith(".csv") else file[:-8]
            model_file = os.path.join(model_folder, name)
            
            if (os.path.isfile(model_file)): # query the user if overwrite file
                if (not yes_to_all and overwrite_existing_files == None):
                    if (is_python_3):
                        response = input("overwrite existing model files? y/N")
                    else:
                        response = raw_input("overwrite existing model files? y/N")
                    if (not response in ["no", "No", "NO", "n", "N"]):
                        overwrite_existing_files = True
                    else:
                        overwrite_existing_files = False
                if (yes_to_all or overwrite_existing_files):
                    os.remove(model_file)
                else:
                    print("file already exists", model_file)
                    sys.exit(1)
            
            print("training model from file ", curr_file)
            learned_model = LearnRules(training_file=curr_file, modelType=model_type, njobs=1, remove_duplicate_features=not keep_duplicate_features, take_max_for_duplicates=not mean_over_duplicates)
            if (not learned_model.is_empty):
                print("writing model to file ", model_file, end="")
                learned_model.saveToDisk(model_file)
                print(" .. done")
            print()
    

def train_wrapper(model, lab_configs):
    training_set_folder = "../caldera/training-data/goodops-useful-num10k/training/"
    model_folder_prefix = "../caldera/trained-models/goodops-useful-num10k-test/trained_model_"
        
    CLASSIFIERS = set(["LOGR", "LOGRCV", "RF", "SVCCV", "SVC", "DT", "KNN", "KRNCV_RG"])        
        
    print("Training %s model" % model)
    if (not model in CLASSIFIERS): # is regressor
        train_model(model_folder_prefix + model.lower() + "_aggmean/", training_set_folder, model, False, True, True)
        lab_configs.append(model.lower() + "_aggmean/")
        train_model(model_folder_prefix + model.lower() + "_aggmax/", training_set_folder, model, False, False, True)
        lab_configs.append(model.lower() + "_aggmax/")
    else:
        train_model(model_folder_prefix + model.lower() + "/", training_set_folder, model, False, False, True)
        lab_configs.append(model.lower() + "/")
    print()



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--manual", action="store_true", required=False, help="adapt options in script directly")
    
    argparser.add_argument("--training-set-folder", type=str, required=False, help="path to training set files (must be *.csv, where last column is the class)")
    argparser.add_argument("--model-folder", type=str, required=False, help="path to folder where to store model files in")
    argparser.add_argument("--model-type", type=str, required=False, help="the type of the learning model: can be one of 'LRCV', 'LG', 'RF' , 'SVMCV','NBB', 'NBG', 'DT'")
    argparser.add_argument("--keep-duplicate-features", action="store_true", required=False, help="elimination and aggregation of duplicate feature vectors, default is eliminate")
    argparser.add_argument("--mean-over-duplicates", action="store_true", required=False, help="aggregating eliminated duplicate feature vectors by taking max or mean (default is max)")
    
    args = argparser.parse_args()
    
    if (not args.manual):
        if (not args.training_set_folder or not args.model_folder or not args.model_type):
            print("INPUT ERROR: the following three options need to be provided: --training-set-folder --model-folder --model-type")
            sys.exit(1)
        train_model(args.model_folder, args.training_set_folder, args.model_type, args.keep_duplicate_features, args.mean_over_duplicates)
    else:
        # possible models (still not a complete list): ["LOGR", "LINR", "RF", "SVC", "SVR", "DT", "DT_RG", "DTGD_RG", "KNN", "KNN_R", "RF_RG", "RFGD_RG", "SVCCV", "SVRGD", "KRNCV_RG", "KRN_RG"]
        
        models = ["LOGR", "SVC", "LINR", "SVR", "DT", "DT_RG", "KRN_RG"]
        
        lab_configs = []
        
        for model in models:
            train_wrapper(model, lab_configs)
            
        print(lab_configs)

    
    
            
            
