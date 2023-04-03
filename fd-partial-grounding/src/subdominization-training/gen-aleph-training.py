#! /usr/bin/env python3

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

from sys import version_info
import sys

is_python_3 = version_info[0] > 2 # test python version

try:
    # Python 3.x
    from builtins import open as file_open
except ImportError:
    # Python 2.x
    from codecs import open as file_open

import parsing_functions
import instantiate

def parse_pddl_file(type, filename):
    try:
        # The builtin open function is shadowed by this module's open function.
        # We use the Latin-1 encoding (which allows a superset of ASCII, of the
        # Latin-* encodings and of UTF-8) to allow special characters in
        # comments. In all other parts, we later validate that only ASCII is
        # used.
        return lisp_parser.parse_nested_list(file_open(filename,
                                                       encoding='ISO-8859-1'))
    except IOError as e:
        raise SystemExit("Error: Could not read file: %s\nReason: %s." %
                         (e.filename, e))
    except lisp_parser.ParseError as e:
        raise SystemExit("Error: Could not parse %s file: %s\nReason: %s." %
                         (type, filename, e))


def write_yap_file(filename_path, filename, probability_class):

    yap_content = """#!/usr/bin/yap -L --    
        :- [aleph].
        :- read_all('{filename}').
        """.format(**locals())

    if probability_class: 
        yap_content += """
        :- set(clauselength, 10).
        :- set(lookahead, 1).
        :- set(evalfn,entropy).
        :- set(mingain, 0.01).
        :- set(prune_tree, false).
        :- set(confidence, 0.001).
        :- induce_tree.
        """
    else:

# The following parameters can affect the size of the search space: i, clauselength, nodes, minpos, minacc, noise, explore, best, openlist, splitvars.
        # set(i,+V): V is a positive integer (default 2). Set upper bound on layers of new variables.
        # set(cache_clauselength,+V): V is a positive integer (default 3). Sets an upper bound on the length of clauses whose coverages are cached for future use.
        # set(nodes,+V) V is a positive integer (default 5000). Set upper bound on the nodes to be explored when searching for an acceptable clause.
        # set(minpos,+V) V is a positive integer (default 1). Set a lower bound on the number of positive examples to be covered by an acceptable clause. If the best clause covers positive examples below this number, then it is not added to the current theory. This can be used to prevent Aleph from adding ground unit clauses to the theory (by setting the value to 2). Beware: you can get counter-intuitive results in conjunction with the minscore setting.
        # set(minposfrac,+V) V is a is a floating point number in the interval [0.0,1.0] (default 0.0). Set a lower bound on the positive examples covered by an acceptable clause as a fraction of the positive examples covered by the head of that clause. If the best clause has a ratio below this number, then it is not added to the current theory. Beware: you can get counter-intuitive results in conjunction with the minpos setting.
        # set(best,+V) V is a `clause label' obtained from an earlier run. This is a list containing at least the number of positives covered, the number of negatives covered, and the length of a clause found on a previous search. Useful when performing searches iteratively.
        # set(explore,+V) V is one of: true or false (default false). If true then forces search to continue until the point that all remaining elements in the search space are definitely worse than the current best element (normally, search would stop when it is certain that all remaining elements are no better than the current best. This is a weaker criterion.) All internal pruning is turned off (see section Built-in and user-defined pruning).
        # set(minacc,+V) V is an floating point number between 0 and 1 (default 0.0). Set a lower bound on the minimum accuracy of an acceptable clause. The accuracy of a clause has the same meaning as precision: that is, it is p/(p+n) where p is the number of positive examples covered by the clause (the true positives) and n is the number of negative examples covered by the clause (the false positives).
       #  set(splitvars,+V) V is one of: true or false (default false). If set to true before constructing a bottom clause, then variable co-references in the bottom clause are split apart by new variables. The new variables can occur at input or output positions of the head literal, and only at output positions in body literals. Equality literals between new and old variables are inserted into the bottom clause to maintain equivalence. It may also result in variable renamed versions of other literals being inserted into the bottom clause. All of this increases the search space considerably and can make the search explore redundant clauses. The current version also elects to perform variable splitting whilst constructing the bottom clause (in contrast to doing it dynamically whilst searching). This was to avoid unnecessary checks that could slow down the search when variable splitting was not required. This means the bottom clause can be extremely large, and the whole process is probably not very practical for large numbers of co-references. The procedure has not been rigourously tested to quantify this.


# The following parameters affect the type of search: search, evalfn, refine, samplesize.
        # set(refine,+V) V is one of: user, auto, or false (default false). Specifies the nature of the customised refinement operator. In all cases, the resulting clauses are required to subsume the bottom clause, if one exists. If false then no customisation is assumed and standard operation results. If user then the user specifies a domain-specific refinement operator with refine/2 statements. If auto then an automatic enumeration of all clauses in the mode language (see section Mode declarations) is performed. The result is a breadth-first branch-and-bound search starting from the empty clause. This is useful if a bottom clause is either not constructed or is constructed lazily. No attempt is made to ensure any kind of optimality and the same clauses may result from several different refinement paths. Some rudimentary checking can be achieved by setting caching to true. The user has to ensure the following for refine is set to auto: (1) the setting to auto is done after the modes and determinations commands, as these are used to generate internally a set of clauses that allow enumeration of clauses in the language; (2) all arguments that are annotated as #T in the modes contain generative definitions for type T. These are called be the clauses generated internally to obtain the appropriate constants; and (3) the head mode is clearly specified using the modeh construct.

    #  set(evalfn,+V): V is one of: coverage, compression, posonly, pbayes, accuracy, laplace, auto_m, mestimate, entropy, gini, sd, wracc, or user (default coverage). Sets the evaluation function for a search. See section Altering the search.
 
    # set(search,+V): V is one of: bf, df, heuristic, ibs, ils, rls, scs id, ic, ar, or false (default bf). Sets the search strategy. If false then no search is performed. See section Altering the search.

   # bf: Enumerates shorter clauses before longer ones. At a given clauselength, clauses are re-ordered based on their evaluation. This is the default search strategy;
        
   # df: Enumerates longer clauses before shorter ones. At a given clauselength, clauses are re-ordered based on their evaluation.
        
   # heuristic: Enumerates clauses in a best-first manner.
        
# ibs Performs an iterative beam search as described by Quinlan and Cameron-Jones, IJCAI-95. Limit set by value for nodes applies to any 1 iteration.
        
 # id Performs an iterative deepening search up to the maximum clause length specified.

    #ils An iterative bf search strategy that, starting from 1, progressively increases the upper-bound on the number of occurrences of a predicate symbol in any clause. Limit set by value for nodes applies to any 1 iteration. This language-based search was developed by Rui Camacho and is described in his PhD thesis.
        
    #rls Use of the GSAT, WSAT, RRR and simulated annealing algorithms for search in ILP. The choice of these is specified by the parameter rls_type (see section Setting Aleph parameters). GSAT, RRR, and annealing all employ random multiple restarts, each of which serves as the starting point for local moves in the search space. A limit on the number of restarts is specified by the parameter tries and that on the number of moves by moves. Annealing is currently restricted to a using a fixed temperature, making it equivalent to an algorithm due to Metropolis. The temperature is specified by setting the parameter temperature. The implementation of WSAT requires a "random-walk probability", which is specified by the parameter walk. A walk probability of 0 is equivalent to GSAT. More details on randomised search can be found in section Randomised search methods.

   #scs A special case of GSAT that results from repeated random selection of clauses from the hypothesis space. The number of clauses is either set by scs_sample or is calculated from the settings for scs_prob and scs_percentile. These represent: the minimum probability of selecting a "good" clause; and the meaning of a "good" clause, namely, that it is in the top K-percentile of clauses. This invokes GSAT search with tries set to the sample size and moves set to 0. Clause selection can either be blind or informed by some preliminary Monte-Carlo style estimation. This is controlled by scs_type. More details can be found in section Randomised search methods.



# The following parameters have an effect on the speed of execution: caching, lazy_negs, proof_strategy, depth, lazy_on_cost, lazy_on_contradiction, searchtime, prooftime.
# The following parameters alter the way things are presented to the user: print, record, portray_hypothesis, portray_search, portray_literals, verbosity,
# The following parameters are concerned with testing theories: test_pos, test_neg, train_pos, train_neg.


        if options.learn_bad:
            minpos, noise = 100, 0
        else:
            minpos, noise = 10, 10
            
        yap_content += """
        :- set(minpos,{minpos}).
        :- set(noise,{noise}).

        :- set(clauselength,5).
        :- set(lookahead,2).
        :- set(search,heuristic).
        :- set(evalfn,coverage).
        :- set(mingain,0).
        :- set(minacc,0).

        :- set(check_redundant,true).
        :- set(check_useless,true).   
        :- set(verbosity,0).
        :- set(print, 10).

        :- induce.
        """.format(**locals())
        

    f = open(filename_path, 'w')
    f.write(yap_content)
    
    os.chmod(filename_path, 0o744)

    f.close()




# use_types is always false at the moment. The problem is that, in problems where there
# are object hierarchies, we need to have some form of "casting" that transforms the types
# above or below the hierarchy. This kind of defeats the purpose of using types because
# the overhead of casting could be larger han any savings by using types.
def get_type(predicate, use_plus, use_types):
    sign = "+" if use_plus else "-"
    typ = predicate.arguments[0].type_name if use_types else "object"
    return f"{sign}'type:{typ}'"

class BackgroundFile:

    def __init__(self, predicates, add_negated_predicates, add_equal_predicate, use_class_probability):
        self.aleph_base_file_content = ""
        self.add_negated_predicates = add_negated_predicates
        self.add_equal_predicate = add_equal_predicate
        self.use_class_probability = use_class_probability
        self.determination_backgrounds = []
        self.aleph_base_file_content = io.StringIO()
        self.aleph_fact_file_content = io.StringIO()
        
        if use_class_probability:
            self.aleph_base_file_content.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
                                          "% specify tree type:\n"
                                          ":- set(tree_type,class_probability).\n"
                                          ":- set(classes,[ground,dont_ground]).\n"
                                          ":- set(dependent,DEPENDENT). % second arg of class is to predicted\n\n")

        predicates = [p for p in predicates if p.name != "="]

        self.aleph_base_file_content.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
                                      "% modes:\n")

        for predicate in predicates:
            arity = len(predicate.arguments)
            for i in range(arity):
                params = "("
                if (arity > 0):
                    params += ", ".join([get_type(predicate.arguments[j], i==j, False) for j in range(arity)]) + ", "
                params += "+task_id)"

                self.aleph_base_file_content.write(":- modeb(*, 'ini:{predicate.name}'{params}).\n".format(**locals()))    
                self.aleph_base_file_content.write(":- modeb(*, 'goal:{predicate.name}'{params}).\n".format(**locals()))
                if (self.add_negated_predicates):
                    self.aleph_base_file_content.write(":- modeb(*, 'ini:not:{predicate.name}'{params}).\n".format(**locals()))    
                    self.aleph_base_file_content.write(":- modeb(*, 'goal:not:{predicate.name}'{params}).\n".format(**locals()))
            self.determination_backgrounds.append("'ini:{name}'/{size}".format(name = predicate.name, size = arity + 1))
            self.determination_backgrounds.append("'goal:{name}'/{size}".format(name = predicate.name, size = arity + 1))
            if (self.add_negated_predicates):
                self.determination_backgrounds.append("'ini:not:{name}'/{size}".format(name = predicate.name, size = arity + 1))
                self.determination_backgrounds.append("'goal:not:{name}'/{size}".format(name = predicate.name, size = arity + 1))

        if (self.add_equal_predicate):
            self.determination_backgrounds.append("equals/3")
            self.determination_backgrounds.append("nequals/3")
            self.aleph_base_file_content.write("\n")
            self.aleph_base_file_content.write(":- modeb(*, equals(+'type:object', +'type:object', +task_id)).\n")
            self.aleph_base_file_content.write(":- modeb(*, notequals(+'type:object', +'type:object', +task_id)).\n")

            self.aleph_base_file_content.write("notequals(A, B, Task):- obj(A, Task), obj(B, Task), not(equals(A, B, Task)).\n")

        self.aleph_base_file_content.write("\n")


    def read_instance(self, task_run, task):
        objects = defaultdict(set)
    
        self.aleph_fact_file_content.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        self.aleph_fact_file_content.write("% init {task_run}\n".format(**locals()))
        for ini_fact in task.init:
            if (type(ini_fact) == pddl.Assign or ini_fact.predicate == "="): # we have our own equality
                continue
            self.aleph_fact_file_content.write("'ini:{ini_fact.predicate}'(".format(**locals()))
            if (len(ini_fact.args) > 0):
                self.aleph_fact_file_content.write("'obj:{ini_fact.args[0]}'".format(**locals()))
                objects[task_run].add(ini_fact.args[0])
                for arg in ini_fact.args[1:]:
                    objects[task_run].add(arg)
                    self.aleph_fact_file_content.write(", 'obj:")
                    self.aleph_fact_file_content.write(arg)
                    self.aleph_fact_file_content.write("'")
                self.aleph_fact_file_content.write(", ")
            self.aleph_fact_file_content.write("'{task_run}').\n".format(**locals()))
            
        self.aleph_fact_file_content.write("\n")
        
        self.aleph_fact_file_content.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        self.aleph_fact_file_content.write("% goal " + task_run + "\n")
        for goal_fact in task.goal.parts:
            self.aleph_fact_file_content.write("'goal:{goal_fact.predicate}'(".format(**locals()))
            if (len(goal_fact.args) > 0):
                self.aleph_fact_file_content.write("'obj:{goal_fact.args[0]}'".format(**locals()))
                objects[task_run].add(goal_fact.args[0])
                for arg in goal_fact.args[1:]:
                    objects[task_run].add(arg)
                    self.aleph_fact_file_content.write(", 'obj:")
                    self.aleph_fact_file_content.write(arg)
                    self.aleph_fact_file_content.write("'")
                self.aleph_fact_file_content.write(", ")
            self.aleph_fact_file_content.write("'" + task_run + "').\n")

        self.aleph_fact_file_content.write("\n")


        
    def write(self, filename, class_parameters):
        with open(filename, "w") as b_file:
            arity = len(class_parameters)
            b_file.write(self.aleph_base_file_content.getvalue().replace("set(dependent,DEPENDENT)", "set(dependent,{D})".format(D=arity + (2 if self.use_class_probability else 1))))
            if (arity > 0):
                params = ", ".join([get_type(class_parameters[j], True, False) for j in range(arity)]) + ", +task_id"
            else:
                params = "+task_id"

            
            if (self.use_class_probability):
                b_file.write(":- modeh(1, class({params}, -class)).\n\n".format(**locals()))
            else:
                b_file.write(":- modeh(1, class({params})).\n\n".format(**locals()))
            
            b_file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
            b_file.write("% determinations:\n")
            for bg in self.determination_backgrounds:
                b_file.write(":- determination(class/{arity}, {pred}).\n".format(arity=len(class_parameters) + (2 if self.use_class_probability else 1), pred=bg))
                
            b_file.write("\n")
            
            if self.add_negated_predicates:
                b_file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
                b_file.write("% negated predicates:\n")
                for predicate in predicates:
                    if (len(predicate.arguments) > 26):
                        sys.exit("lazy programmer ERROR")
                    args = [x for x in string.ascii_uppercase[:len(predicate.arguments)]]
                    args_string = args[0]
                    for arg in args[1:]:
                        args_string += ", " + arg
                    b_file.write("'ini:not:{predicate.name}'({args_string}, Task):- ".format(**locals()))
                    b_file.write("obj({args[0]}, Task)".format(**locals()))
                    for arg in args[1:]:
                        b_file.write(", obj({arg}, Task)".format(**locals()))
                    b_file.write(", not('ini:{predicate.name}'({args_string}, Task)).\n".format(**locals()))
                    b_file.write("'goal:not:{predicate.name}'({args_string}, Task):- ".format(**locals()))
                    b_file.write("obj({args[0]}, Task)".format(**locals()))
                    for arg in args[1:]:
                        b_file.write(", obj({arg}, Task)".format(**locals()))
                    b_file.write(", not('goal:{predicate.name}'({args_string}, Task)).\n".format(**locals()))
                
                b_file.write("\n")
                
                for task in objects.keys():
                    b_file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
                    b_file.write("% all objects of task {task}:\n".format(**locals()))
                    for obj in objects[task]:
                        b_file.write("obj('obj:{obj}', '{task}').\n".format(**locals()))
                    b_file.write("\n")
                    
            if self.add_equal_predicate:                
                for task in objects.keys():
                    b_file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
                    b_file.write("% equals task {task}:\n".format(**locals()))
                    for obj in objects[task]:
                        b_file.write("equals('obj:{obj}', 'obj:{obj}', '{task}').\n".format(**locals()))
                    b_file.write("\n")
                
                b_file.write("\n")
                
            b_file.write(self.aleph_fact_file_content.getvalue())

        # self.aleph_base_file_content.close()
        # self.aleph_fact_file_content.close()


def write_examples_file(filename, examples):
    with open(filename, "w") as n_file:
        n_file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        n_file.write("% training data:\n")
        for task in examples:
            for arguments in examples[task]:
                n_file.write(get_class_example(arguments, task))


def write_class_probability_examples_file(filename, good_examples, bad_examples):
    with open(filename, "w") as f_file:
        f_file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        f_file.write("% training data")
            
        for task in set(good_examples.keys()).union(set(bad_examples.keys())):
            for arguments in good_examples[task]:
                f_file.write("class('obj:{a}'".format(a=arguments[0]))
                for arg in arguments[1:]:
                    f_file.write(", 'obj:{arg}'".format(**locals()))
                if (options.class_probability):
                    f_file.write(", '{task}', ground).\n".format(**locals()))
                else:
                    f_file.write(", '{task}').\n".format(**locals()))

            for arguments in bad_examples[task]:
                f_file.write("class('obj:{a}'".format(a=arguments[0]))
                for arg in arguments[1:]:
                    f_file.write(", 'obj:{arg}'".format(**locals()))
                f_file.write(", '{task}', dont_ground).\n".format(**locals()))


def get_arg_list(action):
    return action.split("(")[1].replace(")", "").split(",")

def get_class_example(arguments, task):

    result = "class('obj:{a}'".format(a=arguments[0])
    for arg in arguments[1:]:
        result += ", 'obj:{arg}'".format(**locals())
    result += ", '{task}').\n".format(**locals())
    return result
                        
    

if __name__ == "__main__":
    import argparse
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("runs_folder", help="path to task pddl file")
    argparser.add_argument("store_training_data", help="Directory to store the training data by gen-subdominization-training")    
    argparser.add_argument("--op-file", default="sas_plan", help="Filename to determine whether to learn from plans (sas_plan) or the set of optimal_operators")
    argparser.add_argument("--all-ops-file", default="all_operators.bz2", help="specify a file with the conditional rule")
    argparser.add_argument("--domain-name", default="domain", help="name of the domain")
    argparser.add_argument("--class-probability", action="store_true", help="write files for class probability, otherwise good/bad files for actions")
    argparser.add_argument("--learn-bad", action="store_true", help="switches good/bad files for actions")
    argparser.add_argument("--add-negated-predicates", action="store_true", help="add negation to model")
    argparser.add_argument("--add-equal-predicate", action="store_true", help="add new equal predicate")
    argparser.add_argument("--conditional_rule", help="specify a file with the conditional rule")
    
    
    options = argparser.parse_args()

    if options.conditional_rule and not os.path.exists(options.conditional_rule):
        print ("File does not exist: ", options.conditional_rule)
        sys.exit()
        
    if os.path.exists(options.store_training_data):
        if (is_python_3):
            result = input('Output path "{}" already exists. Overwrite (Y/n)?'.format(options.store_training_data))
        else:
            result = raw_input('Output path "{}" already exists. Overwrite (Y/n)?'.format(options.store_training_data))
        if result.lower() not in ['y', 'yes', '']:
            sys.exit()
        shutil.rmtree(options.store_training_data)

    operators_filename = options.op_file

    if not os.path.exists(options.store_training_data):
        os.makedirs(options.store_training_data)
        shutil.copy ("aleph/aleph.pl", options.store_training_data)
        
    if options.conditional_rule:
        with open(options.conditional_rule, "r") as conditional_rule_file:
            content = conditional_rule_file.readlines()
            rule_spec, lines = content[0], content[1:]

            action1 = rule_spec.split("=>")[0].strip().split(" ")[-1]
            action2 = rule_spec.split("=>")[1].strip()

            args1, args2 = get_arg_list(action1), get_arg_list(action2), 
            
            arguments = sorted(set(args1 + args2))
            arg_pos = []
            for a in arguments:
                if a in args1:
                    arg_pos.append((0, args1.index(a)))
                else:
                    arg_pos.append((1, args2.index(a)))

            print (arg_pos)
                    
            print (action1, action2, arguments)
            good_examples = defaultdict(list)
            bad_examples = defaultdict(list)
            tasks_examples = set()
            for line in lines:
                 # print (line, end='')
                 sign = line[0]
                 task = line.split(":")[0][2:].strip()
                 
                 acts = line.strip().split(" ")[-2:]
                 acts_args = list(map(get_arg_list, acts))
                 
                 combined_args = [acts_args[w][pos] for w, pos in arg_pos]

                 tasks_examples.add (task)
                 
                 if sign == "+":
                     good_examples[task].append(combined_args)
                 else:
                     assert sign == "-"
                     bad_examples[task].append(combined_args)

            # for task in tasks:
            #     print (task, len(good_examples[task]), len(bad_examples[task]))
            
            filename_path = os.path.join(options.store_training_data, "combined_rules")
            # background_file.write(filename_path + ".b", schema.parameters)

            # write_examples_file(filename_path + ".f", good_examples)
            # write_examples_file(filename_path + ".n", bad_examples)

            all_instances = sorted([d for d in os.listdir(options.runs_folder) if os.path.isfile('{}/{}/{}'.format(options.runs_folder, d, operators_filename))])
            print (all_instances)
            domain_filename = '{}/{}/{}'.format(options.runs_folder, all_instances[0], "domain.pddl")
            domain_pddl = parse_pddl_file("domain", domain_filename)
            domain_name, domain_requirements, types, type_dict, constants, predicates, predicate_dict, functions, action_schemas, axioms = parsing_functions.parse_domain_pddl(domain_pddl)

            bg_file = BackgroundFile(predicates, options.add_negated_predicates, options.add_equal_predicate, options.class_probability)

            # handle the training instances
            for task_run in all_instances:
                if not task_run in tasks_examples:
                    continue

                print ("Processing ", task_run)
                
                domain_filename = '{}/{}/{}'.format(options.runs_folder, task_run, "domain.pddl")
                task_filename = '{}/{}/{}'.format(options.runs_folder, task_run, "problem.pddl")
                plan_filename = '{}/{}/{}'.format(options.runs_folder, task_run, operators_filename)

                domain_pddl = parse_pddl_file("domain", domain_filename)
                task_pddl = parse_pddl_file("task", task_filename)

                task = parsing_functions.parse_task(domain_pddl, task_pddl)

                bg_file.read_instance(task_run, task)


            filename = "combined_rules"
            write_yap_file (os.path.join(options.store_training_data, "learn-" + filename), filename, options.class_probability)

            bg_file.write(filename_path + ".b", arguments)
            
            write_examples_file(filename_path + ".f", good_examples)
            write_examples_file(filename_path + ".n", bad_examples)


        sys.exit(0)


    all_instances = sorted([d for d in os.listdir(options.runs_folder) if os.path.isfile('{}/{}/{}'.format(options.runs_folder, d, operators_filename))])
    print (all_instances)
    domain_filename = '{}/{}/{}'.format(options.runs_folder, all_instances[0], "domain.pddl")
    domain_pddl = parse_pddl_file("domain", domain_filename)
    domain_name, domain_requirements, types, type_dict, constants, predicates, predicate_dict, functions, action_schemas, axioms = parsing_functions.parse_domain_pddl(domain_pddl)

    bg_file = BackgroundFile(predicates, options.add_negated_predicates, options.add_equal_predicate, options.class_probability)
        
    # handle the training instances
    good_operators = defaultdict(lambda : defaultdict(list))
    bad_operators = defaultdict(lambda : defaultdict(list))
    

    for task_run in all_instances:
        print ("Processing ", task_run)
        domain_filename = '{}/{}/{}'.format(options.runs_folder, task_run, "domain.pddl")
        task_filename = '{}/{}/{}'.format(options.runs_folder, task_run, "problem.pddl")
        plan_filename = '{}/{}/{}'.format(options.runs_folder, task_run, operators_filename)
 
        domain_pddl = parse_pddl_file("domain", domain_filename)
        task_pddl = parse_pddl_file("task", task_filename)
 
        all_operators_filename = '{}/{}/{}'.format(options.runs_folder, task_run, options.all_ops_file)
            
        task = parsing_functions.parse_task(domain_pddl, task_pddl)

        bg_file.read_instance(task_run, task)

        with open(plan_filename, "r") as plan_file:
            plan = set(map(lambda x : tuple(x.replace("\n", "").replace(")", "").replace("(", "").split(" ")), plan_file.readlines()))
            # Write good and bad operators 
            with bz2.BZ2File(all_operators_filename, "r") as actions:
                for a in actions:
                    action = a.decode("utf-8")
                    if "(" in action: 
                        schema, arguments = action.split("(")
                        arguments = [x.strip() for x in arguments.strip()[:-1].split(",")]
                    else:
                        tmp = [x.strip() for x in action.split(" ")]
                        schema = tmp[0]
                        arguments = tmp[1:]
                    if (tuple([schema] + arguments) in plan): # is in plan
                        good_operators[schema][task_run].append(arguments)
                    else:
                        bad_operators[schema][task_run].append(arguments)

            
    # write the actual files
    for schema in action_schemas:
        filename = schema.name
        filename_path = os.path.join(options.store_training_data, filename)

        write_yap_file (os.path.join(options.store_training_data, "learn-" + filename), filename, options.class_probability)

        bg_file.write(filename_path + ".b", [param.type_name for param in schema.parameters])

        if options.learn_bad:
            good_operators, bad_operators = bad_operators, good_operators

        
        if (not options.class_probability):
            write_examples_file(filename_path + ".f", good_operators[schema.name])
            write_examples_file(filename_path + ".n", bad_operators[schema.name])
        else:
            write_class_probability_examples_file(filename_path + ".f", good_operators[schema.name], bad_operators[schema.name])
            
            
