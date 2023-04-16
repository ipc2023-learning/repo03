#! /usr/bin/env python

import os
import io
import time
import locale
import bz2
import sys
import pandas
import numpy

from sys import version_info

is_python_3 = version_info[0] > 2 # test python version



class Timer(object):
    def __init__(self):
        self.start_time = time.time()
        self.start_clock = self._clock()

    def _clock(self):
        times = os.times()
        return times[0] + times[1]

    def __str__(self):
        return "[%.3fs CPU, %.3fs wall-clock]" % (
            self._clock() - self.start_clock,
            time.time() - self.start_time)

def remove_duplicates(dataset, take_max): # don't use this, any more.. super slow
    grouped = dataset.groupby([x for x in range(dataset.shape[1] - 1)])    
    filtered_dataset = pandas.DataFrame(columns=[x for x in range(dataset.shape[1])])
    
    timer = Timer()
        
    i = 0
    if (take_max):
        aggregate = grouped.max()
    else:
        aggregate = grouped.mean()
        
    for feature, entry in aggregate.iterrows():
        l = list(feature) # the feature vector
        l.append(entry.get_values()[0]) # the mean outcome for this feature
        filtered_dataset.loc[i] = l
        i += 1
        
    print("removing duplicates reduced size of training set from %d to %d, (aggregating using %s) %s" % (dataset.shape[0], filtered_dataset.shape[0], ("max" if take_max else "mean"), timer))
    return filtered_dataset
    
def remove_duplicates_optimized_and_write_to_file(dataset, take_max, output_file):        
    timer = Timer()
    
    dataset.sort()
    
    reduced_size = 1
    first_line = dataset[0]
    if (len(first_line) <= 3):
        output_file.write("0")
        return
    prev_features = first_line[:-2]
    aggregate = []
    
    for line in dataset:
        label = line[-2:-1]
        features = line[:-2]
        if (prev_features != features):
            reduced_size += 1
            agg = max(aggregate) if take_max else numpy.mean(aggregate)
            output_file.write(prev_features + str(agg) + "\n")
            aggregate = []
            prev_features = features
        aggregate.append(int(label))
    agg = max(aggregate) if take_max else numpy.mean(aggregate)
    output_file.write(prev_features + str(agg) + "\n")
    
    print("removing duplicates reduced size of training set from %d to %d, (aggregating using %s) %s" % (len(dataset), reduced_size, ("max" if take_max else "mean"), timer))
    
    
def get_dataset_from_csv(csv_file, keep_duplicate_features = False, aggregate_max = True):
    if (not os.path.isfile(csv_file)):
        print("ERROR: file does not exist: %s" % csv_file)
        sys.exit(1)
    if (csv_file.endswith(".csv") or csv_file.endswith(".csv.bz2")):
        if (csv_file.endswith(".csv.bz2")):
            file = bz2.open(csv_file, "rt")
        else:
            file = open(csv_file, "rt")
        
        if (keep_duplicate_features):
            dataset= pandas.read_csv(file, sep=",", header=None)
            file.close()
            if (dataset.shape[1] == 1): # no rules for this schema
                print("action schema %s skipped, no rules for this schema" % csv_file)
                return None
            return dataset
        else:
            tmp_file = io.StringIO()
            remove_duplicates_optimized_and_write_to_file(file.readlines(), aggregate_max, tmp_file)
            file.close()
            tmp_file.seek(0)
            dataset = pandas.read_csv(tmp_file, sep=",", header=None)
            tmp_file.close()
            if (dataset.shape[1] == 1): # no rules for this schema
                print("action schema %s skipped, no rules for this schema" % csv_file)
                return None
            return dataset
    else:
        print("ERROR: unsupported file format: %s" % csv_file)
        sys.exit(1)
        
        
        
def write_latex_file(schemas, models, data_path_prefix, file):
    if (os.path.isfile(file)): # query the user if overwrite file
        if (is_python_3):
            response = input("overwrite existing file? (" + file + ") y/N ")
        else:                
            response = raw_input("overwrite existing file? (" + file + ") y/N ")
        if (response in ["no", "No", "NO", "n", "N"]):
            print("file already exists", file)
            sys.exit(1)
        else:
            os.remove(file)
            
    with open(file, "w") as outfile:
        outfile.write("""\\documentclass{article}\n\n"""
                   """\\usepackage{filecontents}\n"""
                   """\\usepackage{pgfplots, pgfplotstable}\n"""
                   """\\usepgfplotslibrary{statistics,external}\n"""
                   """\n"""
                   """\\tikzexternalize[prefix=TMP/]\n"""
                   """\n"""
#                    """\\usepackage{listings}\n""" # didnt work
#                    """\\newcommand{\\codelst}{\\begingroup\n"""
#                    """  \\catcode`_=12 \\docodelst}\n"""
#                    """\\newcommand{\\docodelst}[1]{%\n"""
#                    """  \\lstinputlisting[caption=\\texttt{#1}]{#1}%\n"""
#                    """  \\endgroup\n"""
#                    """}\n\n"""
                   """\\input{plot_macro}\n"""
                   """\n"""
                   """\\begin{document}\n\n""")
        for model in models:
            outfile.write("\\section{" + model.replace("_", "-") + "}\n\n")
            outfile.write("training set left, validation set right\n\n")
            for schema in schemas[model]:
                stripped_schema = schema.replace("_", "-")
                prefix = os.path.join(data_path_prefix, model, "training/")
                outfile.write("\histplot{{{prefix}}}{{{schema}}}{{{stripped_schema}}}\n".format(**locals()))
                prefix = os.path.join(data_path_prefix, model, "testing/")
                outfile.write("\histplot{{{prefix}}}{{{schema}}}{{{stripped_schema}}}\n\n".format(**locals()))
            prefix = os.path.join(data_path_prefix, model, "training/")
            outfile.write("\histplot{{{prefix}}}{{all}}{{all}}\n".format(**locals()))
            prefix = os.path.join(data_path_prefix, model, "testing/")
            outfile.write("\histplot{{{prefix}}}{{all}}{{all}}\n\n".format(**locals()))
                
        outfile.write("\\end{document}\n")

def write_samples_to_file(samples, file, overwrite = False):
    if (os.path.isfile(file)): # query the user if overwrite file
        if (not overwrite):
            if (is_python_3):
                response = input("overwrite existing file? (" + file + ") y/N")
            else:
                response = raw_input("overwrite existing file? (" + file + ") y/N")
            if (response in ["no", "No", "NO", "n", "N"]):
                overwrite = False
            else:
                overwrite = True
        if (overwrite):
            os.remove(file)
        else:
            print("file already exists", file)
            sys.exit(1)
    if (not os.path.isdir(os.path.dirname(file))):
        os.makedirs(os.path.dirname(file))
    with open(file, "w") as outfile:
        for sample in samples:
            outfile.write(str(round(sample, 2)))
            outfile.write("\n")
