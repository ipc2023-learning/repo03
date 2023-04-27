import argparse
import sys


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "domain", help="path to domain pddl file")
    argparser.add_argument(
        "task", help="path to task pddl file")
    argparser.add_argument(
        "--relaxed", dest="generate_relaxed_task", action="store_true",
        help="output relaxed task (no delete effects)")
    argparser.add_argument(
        "--full-encoding",
        dest="use_partial_encoding", action="store_false",
        help="By default we represent facts that occur in multiple "
        "mutex groups only in one variable. Using this parameter adds "
        "these facts to multiple variables. This can make the meaning "
        "of the variables clearer, but increases the number of facts.")
    argparser.add_argument(
        "--invariant-generation-max-candidates", default=100000, type=int,
        help="max number of candidates for invariant generation "
        "(default: %(default)d). Set to 0 to disable invariant "
        "generation and obtain only binary variables. The limit is "
        "needed for grounded input files that would otherwise produce "
        "too many candidates.")
    argparser.add_argument(
        "--sas-file", default="output.sas",
        help="path to the SAS output file (default: %(default)s)")
    argparser.add_argument(
        "--invariant-generation-max-time", default=300, type=int,
        help="max time for invariant generation (default: %(default)ds)")
    argparser.add_argument(
        "--add-implied-preconditions", action="store_true",
        help="infer additional preconditions. This setting can cause a "
        "severe performance penalty due to weaker relevance analysis "
        "(see issue7).")
    argparser.add_argument(
        "--keep-unreachable-facts",
        dest="filter_unreachable_facts", action="store_false",
        help="keep facts that can't be reached from the initial state")
    argparser.add_argument(
        "--skip-variable-reordering",
        dest="reorder_variables", action="store_false",
        help="do not reorder variables based on the causal graph. Do not use "
        "this option with the causal graph heuristic!")
    argparser.add_argument(
        "--keep-unimportant-variables",
        dest="filter_unimportant_vars", action="store_false",
        help="keep variables that do not influence the goal in the causal graph")
    argparser.add_argument(
        "--dump-task", action="store_true",
        help="dump human-readable SAS+ representation of the task")
    argparser.add_argument(
        "--layer-strategy", default="min", choices=["min", "max"],
        help="How to assign layers to derived variables. 'min' attempts to put as "
        "many variables into the same layer as possible, while 'max' puts each variable "
        "into its own layer unless it is part of a cycle.")
    argparser.add_argument(
        "--grounding-action-queue-ordering", default="fifo", type=str,
        help="type of queue that is used in compute_model to order the actions (default: %(default)d)")
    argparser.add_argument(
        "--trained-model-folder", type=str,
        help="The folder that should contain the trained model and relevant files if 'trained' is used as queue ordering")
    argparser.add_argument(
        "--aleph-model-file", type=str,
        help="The file that contains the model trained by aleph. 'aleph' has to be used as --grounding-action-queue-ordering")
    argparser.add_argument(
        "--termination-condition", type=str, default=["default"], nargs="+", # at least one argument must be given
        help="the termination condition, which may be followed by additional arguments used by the respective condition")
    argparser.add_argument(
        "--hard-rules", type=str, nargs=2,
        help="type of hard rule evaluator and file that contains the rules.")
    argparser.add_argument(
        "--actions-file", type=str, nargs=1,
        help="file that contains (line by line) the actions that should be grounded.")
    argparser.add_argument(
        "--reachable-actions-output-file", type=str, nargs=1,
        help="file to write the actions to that were reachable during grounding.")
    argparser.add_argument(
        "--action-schema-ratios", type=str,
        help="The file that contains the ratios over the total number of actions of a schema in which it occurs in a plan.")
    argparser.add_argument(
        "--plan-ratios", action="store_true",
        help="Only has an effect in combination with the --action-schema-ratios option. If provided, the ratios are assumed "
             "to be for each actions schema the ratio of the number of actions of that schema in a plan over the number of "
             "total actions in the plan.")
    argparser.add_argument(
        "--batch-evaluation", action="store_true",
        help="When using a trained model, the action evaluation is done in batches.")
    argparser.add_argument(
        "--relaxed-plan-file", type=str,
        help="Path of a file that contains the facts achieved in a relaxed plan." 
             "Enables support for \"split\" rules in the rule evaluator.")
    argparser.add_argument(
        "--policy-file", type=str,
        help="Path of a file that contains a policy that should be used to guide grounding.")
    argparser.add_argument(
        "--fasttext-model", type=str,
        help="Path of a file that contains the trained fasttest model.")
    argparser.add_argument(
        "--random-seed", type=int,
        help="Only used for random action sorting so far.")
    argparser.add_argument(
        "--optimized-fasttext", action="store_true",
        help="Optimized evaluation for fasttext models.")
    argparser.add_argument(
        "--ignore-bad-actions", action="store_true",
        help="Completely ignore actions evaluated as bad by the GoodBadRuleEvaluator.")

    return argparser.parse_args()


def copy_args_to_module(args):
    module_dict = sys.modules[__name__].__dict__
    for key, value in vars(args).items():
        module_dict[key] = value


def setup():
    args = parse_args()
    copy_args_to_module(args)


setup()
