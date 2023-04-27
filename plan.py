#! /usr/bin/env python

from __future__ import print_function

import argparse
import json
import os.path
import re
import shutil
import subprocess
import tarfile
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PLAN_PARTIAL_GROUNDING = os.path.join(ROOT, "plan-partial-grounding.py")
FD_PARTIAL_GROUNDING = os.path.join(ROOT, "fd-partial-grounding", "fast-downward.py")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain_knowledge", help="path to domain knowledge file")
    parser.add_argument("domain", help="path to domain file")
    parser.add_argument("problem", help="path to problem file")
    parser.add_argument("plan", help="path to output plan file")
    return parser.parse_args()


def get_lama(bound):
    return [
        "--if-unit-cost",
        "--evaluator",
        "hlm=lmcount(lm_reasonable_orders_hps(lm_rhw()),pref=false)",
        "--evaluator", "hff=ff()",
        "--search", f"""iterated([
                         lazy_greedy([hff,hlm],preferred=[hff,hlm]),
                         lazy_wastar([hff,hlm],preferred=[hff,hlm],w=5),
                         lazy_wastar([hff,hlm],preferred=[hff,hlm],w=3),
                         lazy_wastar([hff,hlm],preferred=[hff,hlm],w=2),
                         lazy_wastar([hff,hlm],preferred=[hff,hlm],w=1),
                         ],repeat_last=true,continue_on_fail=true, bound={bound})""",
        "--if-non-unit-cost",
        "--evaluator",
        "hlm1=lmcount(lm_reasonable_orders_hps(lm_rhw()),transform=adapt_costs(one),pref=false)",
        "--evaluator", "hff1=ff(transform=adapt_costs(one))",
        "--evaluator",
        "hlm2=lmcount(lm_reasonable_orders_hps(lm_rhw()),transform=adapt_costs(plusone),pref=false)",
        "--evaluator", "hff2=ff(transform=adapt_costs(plusone))",
        "--search", f"""iterated([
                         lazy_greedy([hff1,hlm1],preferred=[hff1,hlm1],
                                     cost_type=one,reopen_closed=false),
                         lazy_greedy([hff2,hlm2],preferred=[hff2,hlm2],
                                     reopen_closed=false),
                         lazy_wastar([hff2,hlm2],preferred=[hff2,hlm2],w=5),
                         lazy_wastar([hff2,hlm2],preferred=[hff2,hlm2],w=3),
                         lazy_wastar([hff2,hlm2],preferred=[hff2,hlm2],w=2),
                         lazy_wastar([hff2,hlm2],preferred=[hff2,hlm2],w=1)
                         ],repeat_last=true,continue_on_fail=true, bound={bound})""",
        # Append --always to be on the safe side if we want to append
        # additional options later.
        "--always"]


def get_options(config_file):
    with open(config_file, "r") as cfile:
        config_dict = json.load(cfile)

    config = ["--alias", config_dict["alias"], "--h2-preprocessor"]

    if "queue_type" in config_dict:
        config += ["--grounding-queue", config_dict["queue_type"],
                   ]

        config += ["--incremental-grounding"]
        # TODO add incremental grounding options to config file and include them here

    if "ignore_bad_actions" in config_dict and config_dict["ignore_bad_actions"].lower() == "true":
        config += ["--ignore-bad-actions"]

    # TODO add termination condition

    return config


def main():
    args = parse_args()

    dk_folder = f"{os.path.basename(args.domain_knowledge)}-extracted"

    # uncompress domain knowledge file
    with tarfile.open(args.domain_knowledge, "r:gz") as tar:
        if os.path.exists(dk_folder):
            shutil.rmtree(dk_folder)
        tar.extractall(dk_folder)

    config = get_options(os.path.join(dk_folder, "config"))

    # TODO check if partial grounding should be run at all
    # TODO add time + memory limits
    # TODO maybe limit time for partial grounding translator?
    subprocess.run([sys.executable, PLAN_PARTIAL_GROUNDING] +
                   [dk_folder, args.domain, args.problem, "--plan", args.plan] +
                   config)

    # TODO add memory limit
    # run standard LAMA as fallback or to improve found solution
    lama_config = ["--transform-task", f"{ROOT}/fd-partial-grounding/builds/release/bin/preprocess-h2",
                   "--transform-task-options", "h2_time_limit,300",
                   "--plan", args.plan, args.domain, args.problem]
    if os.path.isfile(args.plan):
        with open(args.plan) as plan_file:
            cost_line = plan_file.readlines()[-1]
            cost_re = re.compile(r'^; cost = (\d+)')
            bound = cost_re.match(cost_line).group(1)

        lama_config += get_lama(bound)
    else:
        lama_config = ["--alias", "lama"] + lama_config

    subprocess.run([sys.executable, FD_PARTIAL_GROUNDING] + lama_config)


if __name__ == "__main__":
    main()
