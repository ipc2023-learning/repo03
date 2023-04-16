#! /usr/bin/env python

from __future__ import print_function

import argparse
import os.path
import subprocess
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
FD_PARTIAL_GROUNDING = os.path.join(ROOT, "fd-partial-grounding")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain_knowledge", help="path to domain knowledge file")
    parser.add_argument("domain", help="path to domain file")
    parser.add_argument("problem", help="path to problem file")
    parser.add_argument("--plan", default=None, help="path to output plan file")
    parser.add_argument("--alias", default="lama-first", type=str,
                        help="alias for the search config")

    parser.add_argument("--h2-preprocessor", action="store_true",
                        help="run h2 preprocessor")
    parser.add_argument("--h2-time-limit", type=int, default=300,
                        help="time limit for h2 preprocessor, default 5min")

    parser.add_argument("--incremental-grounding", action="store_true")
    parser.add_argument("--incremental-grounding-search-time-limit", type=int, default=600,
                        help="search time limit in seconds per iteration")
    parser.add_argument("--incremental-grounding-minimum", type=int,
                        help="minimum number of actions grounded in the first iteration")
    parser.add_argument("--incremental-grounding-increment", type=int,
                        help="absolute increment in number of actions grounded per iteration")
    parser.add_argument("--incremental-grounding-increment-percentage", type=int,
                        help="relative increment in number of actions grounded per iteration,"
                             "e.g. 10 for 10% additional actions. If provided in combination with "
                             "--incremental-grounding-increment, the maximum of the two is taken.")

    # TODO introduce plan-ratios queues
    parser.add_argument("--grounding-queue", type=str, default="trained",
                        help="Options are trained/roundrobintrained (sklearn model), "
                             "aleph/roundrobinaleph (aleph model)),"
                             "noveltyfifo/roundrobinnovelty (novelty)")

    return parser.parse_args()


def main():
    args = parse_args()
    driver_options = ["--alias", args.alias]

    if args.plan:
        driver_options += ["--plan-file", args.plan]
    if args.h2_preprocessor:
        driver_options += ["--transform-task", "fd-partial-grounding/builds/release/bin/preprocess-h2",
                           "--transform-task-options", f"h2_time_limit,{args.h2_time_limit}"]

    translate_options = ["--translate-options",
                         "--batch-evaluation",
                         "--grounding-action-queue-ordering",
                         args.grounding_queue]
    if args.grounding_queue in ["trained", "roundrobintrained"]:
        translate_options += ["--trained-model-folder", args.domain_knowledge]
    elif args.grounding_queue in ["aleph", "roundrobinaleph"]:
        # TODO this does not work yet, need to provide the actual file
        translate_options += ["--aleph-model-file", args.domain_knowledge]

    if args.incremental_grounding:
        driver_options += ["--incremental-grounding",
                           "--incremental-grounding-search-time-limit", str(args.incremental_grounding_search_time_limit),
                           ]
        if args.incremental_grounding_minimum:
            driver_options += ["--incremental-grounding-minimum", str(args.incremental_grounding_minimum)]
        if args.incremental_grounding_increment:
            driver_options += ["--incremental-grounding-increment", str(args.incremental_grounding_increment)]
        if args.incremental_grounding_increment_percentage:
            driver_options += ["--incremental-grounding-increment-percentage",
                               str(args.incremental_grounding_increment_percentage)]
    else:
        translate_options += ["--termination-condition", "goal-relaxed-reachable"]

    subprocess.check_call([sys.executable, os.path.join(FD_PARTIAL_GROUNDING, "fast-downward.py")] +
                          driver_options +
                          [args.domain, args.problem] +
                          translate_options)


if __name__ == "__main__":
    main()
