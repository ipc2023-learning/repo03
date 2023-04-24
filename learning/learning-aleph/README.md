# Options that are already set by the script (fixed for good/bad actions)
    * minacc: V is an floating point number between 0 and 1 (default 0.0). Set a lower bound on the minimum accuracy of an acceptable clause. The accuracy of a clause has the same meaning as precision: that is, it is p/(p+n) where p is the number of positive examples covered by the clause (the true positives) and n is the number of negative examples covered by the clause (the false positives).


# Options for induce

* induce: Default configuration
* induce_cover:
* induce_max:


# Options for search

The search for individual clauses (when performed) is principally affected by two parameters. One sets the search strategy (search) and the other sets the evaluation function (evalfn).

The following parameters affect the type of search: search, evalfn, refine, samplesize.

* searchtime: Sets an upper bound on the time (in seconds) for a search.

* search: bf, df, heuristic, ibs, ils, rls, scs id, ic, ar, or false (default bf). Sets the search strategy. If false then no search is performed.

    * bf, df (breadth and depth first search)
    * heuristic (gbfs)
    * ibs: iterative beam search (Cameron-Jones). Limit set by value for nodes applies to any 1 iteration.
    * id Performs an iterative deepening search up to the maximum clause length specified.
    * ils An iterative bf search strategy that, starting from 1, progressively increases the upper-bound on the number of occurrences of a predicate symbol in any clause. Limit set by value for nodes applies to any 1 iteration. This language-based search was developed by Rui Camacho and is described in his PhD thesis.
    * rls Use of the GSAT, WSAT, RRR and simulated annealing algorithms for search in ILP. The choice of these is specified by the parameter rls_type. GSAT, RRR, and annealing all employ random multiple restarts, each of which serves as the starting point for local moves in the search space. A limit on the number of restarts is specified by the parameter tries and that on the number of moves by moves. Annealing is currently restricted to a using a fixed temperature, making it equivalent to an algorithm due to Metropolis. The temperature is specified by setting the parameter temperature. The implementation of WSAT requires a "random-walk probability", which is specified by the parameter walk. A walk probability of 0 is equivalent to GSAT.

        * rls_type: gsat, wsat, rrr, or anneal
        * tries: V is a positive integer. Sets the maximum number of restarts allowed for randomised search methods.
        * moves: V is an integer >= 0. Set an upper bound on the number of moves allowed when performing a randomised local search

        * temperature (if rls_type == anneal): Sets the temperature for randomised search using annealing.
        * walk (if rls_type=wsat): V is a value between 0 and 1. It represents the random walk probability for the Walksat algorithm.

    * scs A special case of GSAT that results from repeated random selection of clauses from the hypothesis space. The number of clauses is either set by scs_sample or is calculated from the settings for scs_prob and scs_percentile. These represent: the minimum probability of selecting a "good" clause; and the meaning of a "good" clause, namely, that it is in the top K-percentile of clauses. This invokes GSAT search with tries set to the sample size and moves set to 0. Clause selection can either be blind or informed by some preliminary Monte-Carlo style estimation. This is controlled by scs_type.
        * scs_prob: V is an number in the range [0,1.0). This denotes the minimum probability of obtaining a "good" clause when performing stochastic clause selection.
        * scs_percentile: V is an number in the range (0,100] (usually an integer). This denotes that any clause in the top V-percentile of clauses are considered "good" when performing stochastic clause selection.
        * scs_type: scs_type to be one of blind or informed

* evalfn: V is one of: coverage, compression, posonly, pbayes, accuracy, laplace, auto_m, mestimate, entropy, gini, sd, wracc, or user (default coverage). Sets the evaluation function for a search.
    * m: (evalfn==mestimate)

* samplesize: V is an integer >= 0 (default 0). Sets number of examples selected randomly by the induce or induce_cover commands. The best clause from the sample is added to the theory. A value of 0 turns off random sampling, and the next uncovered example in order of appearance in the file of training examples is selected.


# refinement:

* refine: user, auto, or false (default false).
* set(lookahead,+V)
V is a positive integer. Sets a lookahead value for the automatic refinement operator (obtained by setting refine to auto).



# Specific-to-General search

       :- set(samplesize,4).
       :- set(resample,4).
       :- set(permute_bottom,true).
       :- set(nreduce_bottom,true).



# The following parameters can affect the size of the search space: i, clauselength, nodes, minpos, minacc, noise, explore, best, openlist, splitvars.
    * i: positive integer (default 2). Set upper bound on layers of new variables.
    * nodes: V is a positive integer (default 5000). Set upper bound on the nodes to be explored when searching for an acceptable clause.
    * minpos: V is a positive integer (default 1). Set a lower bound on the number of positive examples to be covered by an acceptable clause. If the best clause covers positive examples below this number, then it is not added to the current theory. This can be used to prevent Aleph from adding ground unit clauses to the theory (by setting the value to 2). Beware: you can get counter-intuitive results in conjunction with the minscore setting.
    * explore: V is one of: true or false (default false). If true then forces search to continue until the point that all remaining elements in the search space are definitely worse than the current best element (normally, search would stop when it is certain that all remaining elements are no better than the current best. This is a weaker criterion.) All internal pruning is turned off.

Not used at the moment:
    * minposfrac: V is a is a floating point number in the interval [0.0,1.0] (default 0.0). Set a lower bound on the positive examples covered by an acceptable clause as a fraction of the positive examples covered by the head of that clause. If the best clause has a ratio below this number, then it is not added to the current theory. Beware: you can get counter-intuitive results in conjunction with the minpos setting.

    * best: V is a `clause label' obtained from an earlier run. This is a list containing at least the number of positives covered, the number of negatives covered, and the length of a clause found on a previous search. Useful when performing searches iteratively.

    * cache_clauselength: is a positive integer (default 3). Sets an upper bound on the length of clauses whose coverages are cached for future use.
    *  splitvars: true or false (default false). If set to true before constructing a bottom clause, then variable co-references in the bottom clause are split apart by new variables. The new variables can occur at input or output positions of the head literal, and only at output positions in body literals. Equality literals between new and old variables are inserted into the bottom clause to maintain equivalence. It may also result in variable renamed versions of other literals being inserted into the bottom clause. All of this increases the search space considerably and can make the search explore redundant clauses. The current version also elects to perform variable splitting whilst constructing the bottom clause (in contrast to doing it dynamically whilst searching). This was to avoid unnecessary checks that could slow down the search when variable splitting was not required. This means the bottom clause can be extremely large, and the whole process is probably not very practical for large numbers of co-references. The procedure has not been rigourously tested to quantify this.


    * newvars: is a positive integer or inf (default inf). Set upper bound on the number of existential variables that can be introduced in the body of a clause.

# The following parameters have an effect on the speed of execution: caching, lazy_negs, proof_strategy, depth, lazy_on_cost, lazy_on_contradiction, searchtime, prooftime.
# The following parameters alter the way things are presented to the user: print, record, portray_hypothesis, portray_search, portray_literals, verbosity,

* print (default 4). Sets an upper bound on the maximum number of literals displayed on any one line of the trace.

# The following parameters are concerned with testing theories: test_pos, test_neg, train_pos, train_neg.


# Other parameters of interest:
* resample:  V is an integer >= 1 or inf (default 1). Sets the number of times an example is resampled when selected by induce/0 or induce_cover/0. That is, is set to some integer N, then the example is repeatedly selected N times by induce/0 or induce_cover/0.



set(permute_bottom,+V)
V is one of: true or false (default false). If true randomly permutes literals in the body of the bottom clause, within the constraints imposed by the mode declarations. The utility of is described by P. Tschorn in Random Local Bottom Clause Permutations for Better Search Space Exploration in Progol-like ILP Systems. (short papers, ILP 2006).
