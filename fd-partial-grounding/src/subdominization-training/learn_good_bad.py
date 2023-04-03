#!/usr/bin/python

import argparse
import os
    
argparser = argparse.ArgumentParser()
argparser.add_argument("domain", help="Directory to store the training data by gen-subdominization-training")

options = argparser.parse_args()

training_dir = '/home/alvaro/projects/subdominization/subdominization_data/{}/training-data-aleph/good-bad/'.format(options.domain)

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
        :- set(lookahead, 2).
        :- set(evalfn,coverage).
        :- set(mingain, 0).
        :- set(minacc,0).
        :- set(minpos, 1000).
        :- set(noise, 0).
        :- set(check_redundant,false).
        :- set(check_useless,false).   
        :- induce.
        """.format(**locals())

        filename_exec = 'aleph/learn-aleph-{}'.format(filename[:-2])

        f = open(filename_exec, 'w')
        f.write(yap_content)
        f.close()
        os.system('chmod +x {}'.format(filename_exec))
    


