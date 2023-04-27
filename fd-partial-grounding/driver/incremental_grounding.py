import logging
import sys

from . import returncodes as rc
from . import run_components
from . import util


def get_new_limit(num_grounded_actions, increment, args):
    if not args.incremental_grounding_increment and args.incremental_grounding_increment_percentage:
        new_limit = int(num_grounded_actions * args.incremental_grounding_increment_percentage)
    else:
        new_limit = int(num_grounded_actions / increment) * increment
        if args.incremental_grounding_increment_percentage:
            inc = max(increment, int(num_grounded_actions * (args.incremental_grounding_increment_percentage - 1)))
            if inc == increment:
                new_limit = int(num_grounded_actions / increment) * increment + increment
            else:
                new_limit = num_grounded_actions + inc
        else:
            new_limit = int(num_grounded_actions / increment) * increment + increment
    return new_limit


def do_incremental_grounding(args):
    if "translate" not in args.components or "search" not in args.components:
        sys.exit("ERROR: need to execute translator and search to do incremental grounding.")

    first_iteration = True

    num_grounded_actions = -1
    num_grounded_actions_last_iteration = -1
    args.search_time_limit = args.incremental_grounding_search_time_limit if args.incremental_grounding_search_time_limit else 10 * 60
    increment = args.incremental_grounding_increment if args.incremental_grounding_increment else 10000
    new_limit = increment
    old_translate_options = list(args.translate_options)
    old_search_options = list(args.search_options)
    
    if args.incremental_grounding_increment_percentage:
        args.incremental_grounding_increment_percentage = args.incremental_grounding_increment_percentage / 100 + 1
        
    while True:
        if args.overall_time_limit and args.overall_time_limit - util.get_elapsed_time() <= 0:
            print("incremental grounding ran out of time")
            sys.exit(rc.TRANSLATE_OUT_OF_TIME)

        if first_iteration and "--termination-condition" in old_translate_options:
            termination_condition = []
            tc_id = old_translate_options.index("--termination-condition")
            for i in range(tc_id, len(old_translate_options)):
                if i > tc_id and old_translate_options[i].startswith("--"):
                    break
                termination_condition.append(old_translate_options[i])
            del old_translate_options[tc_id: i]
        else:
            termination_condition = ["--termination-condition", "goal-relaxed-reachable"]
            if num_grounded_actions != -1:
                new_limit = get_new_limit(num_grounded_actions, increment, args)
                termination_condition += ["min-number", str(new_limit)]
            elif args.incremental_grounding_minimum:
                # TODO does this actually make sense?
                #  this option seems to be obsolete if we allow
                #  passing a termination condition in the first iteration
                termination_condition += ["min-number", str(args.incremental_grounding_minimum),
                                          "percentage", "10",
                                          "max-increment", "20000"]  # TODO make this an option

        first_iteration = False

        args.translate_options = old_translate_options + termination_condition
        
        (exitcode, _, num_grounded_actions) = run_components.run_translate(args, True)

        if num_grounded_actions == num_grounded_actions_last_iteration:
            print("No progress, the same number of actions was grounded in the last iteration. Stopping.")
            break

        num_grounded_actions_last_iteration = num_grounded_actions
        
        print()
        print("translate exit code: {exitcode}".format(**locals()))
        if exitcode in [rc.TRANSLATE_OUT_OF_MEMORY, rc.TRANSLATE_OUT_OF_TIME, rc.TRANSLATE_CRITICAL_ERROR, rc.TRANSLATE_INPUT_ERROR]:
            print("Driver aborting after translator")
            sys.exit(exitcode)
        elif exitcode == rc.TRANSLATE_UNSOLVABLE:
            num_grounded_actions = get_new_limit(num_grounded_actions, increment, args)
            print("Task proved unsolvable in translator, increasing minimum number of grounded actions.")
            continue

        if args.transform_task:
            print()
            run_components.transform_task(args)
        
        args.search_options = list(old_search_options)
        
        (exitcode, _) = run_components.run_search(args)
        
        print()
        print("search exit code: {exitcode}".format(**locals()))
        if exitcode in [rc.SUCCESS, rc.SEARCH_PLAN_FOUND_AND_OUT_OF_MEMORY,
                        rc.SEARCH_PLAN_FOUND_AND_OUT_OF_TIME,
                        rc.SEARCH_PLAN_FOUND_AND_OUT_OF_MEMORY_AND_TIME]:
            break
        elif exitcode in [rc.SEARCH_INPUT_ERROR, rc.SEARCH_UNSUPPORTED]:
            print("Driver aborting after search")
            sys.exit(exitcode)
    if "validate" in args.components:
        (exitcode, _) = run_components.run_validate(args)
        print()
        print("validate exit code: {exitcode}".format(**locals()))
    
    try:
        logging.info(f"Planner time: {util.get_elapsed_time():.2f}s")
    except NotImplementedError:
        # Measuring the runtime of child processes is not supported on Windows.
        pass

    sys.exit(exitcode)
    
