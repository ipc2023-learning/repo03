#!/usr/bin/python

import argparse
import os
    
argparser = argparse.ArgumentParser()
argparser.add_argument("domain", help="Directory to store the training data by gen-subdominization-training")

options = argparser.parse_args()

training_dir = '/home/alvaro/projects/subdominization/subdominization_data/{}/training-data-aleph/class-probability/'.format(options.domain)

if not os.path.exists(training_dir):
    print ("Directory not found: ", training_dir)
    exit()

for filename in  os.listdir(training_dir):
    if filename.endswith(".b"):
        print (filename)
        training_filename = "{}/{}".format(training_dir, filename[:-2])
        
        yap_content = """#!/usr/bin/yap -L --
        
        :- [aleph].
        :- read_all('{training_filename}').
        :- set(clauselength, 10).
        :- set(lookahead, 1).
        :- set(evalfn,entropy).
        :- set(mingain, 0.01).
        :- set(prune_tree, false).
        :- set(confidence, 0.001).
        :- induce_tree.
        """.format(**locals())

        filename_exec = 'aleph/learn-aleph-{}'.format(filename[:-2])

        f = open(filename_exec, 'w')
        f.write(yap_content)
        f.close()
        os.system('chmod +x {}'.format(filename_exec))
    


