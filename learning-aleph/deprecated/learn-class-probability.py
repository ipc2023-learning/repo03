#! /usr/bin/env python

import os


def generate_training_scripts(training_dir, output_dir):
    if not os.path.exists(options.directory):
        print ("Directory not found: ", options.directory)
        return False

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

            filename_exec = '{}/learn-aleph-{}'.format(training_dir,filename[:-2])

            f = open(filename_exec, 'w')
            f.write(yap_content)
            f.close()
            os.system('chmod +x {}'.format(filename_exec))

    return True

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("directory", help="directory where aleph files are and where the results should be stored")
    argparser.add_argument("--output", help="directory where results should be stored (by default this is the same directory as the input files)")

    options = argparser.parse_args()

    generate_training_scripts(options.directory, options.output if options.output else options.directory)
