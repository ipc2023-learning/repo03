import logging
import os
import sys

from . import call
from . import returncodes as rc
from . import util


def compute(args):
    if args.compute_relaxed_info == "powerlifted-facts":
        logging.info("Running powerlifted.")
        
        facts_file = "useful_facts" # TODO make this an option
        
        # take release build by default
        powerlifted = os.path.join(util.REPO_ROOT_DIR, "powerlifted.py")
        
        if not os.path.exists(powerlifted):
            rc.exit_with_driver_input_error(
                "Could not find '{powerlifted}' in builds. "
                "Please run './build.py powerlifted'.".format(**locals()))
        
        try:
            call.check_call(
                "powerlifted",
                [sys.executable] + [powerlifted] + 
                ["-d", args.translate_inputs[0], "-i", args.translate_inputs[1]] + 
                ["-s", "astar", "-e", "print-useful-facts", "-g", "yannakakis", "--useful-facts-file", facts_file],
                time_limit=args.overall_time_limit,
                memory_limit=args.overall_memory_limit)
        except OSError as err:
            rc.exit_with_driver_critical_error(err)
        
        logging.info(f"Successfully computed useful facts with Powerlifted and saved the outcome to {facts_file}. [{util.get_elapsed_time()}s]")
    else:
        logging.error(f"ERROR: unknown --compute-relaxed-info option: {args.compute_relaxed_info}")
        sys.exit(rc.DRIVER_INPUT_ERROR)
