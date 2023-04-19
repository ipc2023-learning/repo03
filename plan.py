#! /usr/bin/env python

from __future__ import print_function

import argparse
import json
import os.path
import shutil
import subprocess
import tarfile
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PLAN_PARTIAL_GROUNDING = os.path.join(ROOT, "plan-partial-grounding.py")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain_knowledge", help="path to domain knowledge file")
    parser.add_argument("domain", help="path to domain file")
    parser.add_argument("problem", help="path to problem file")
    parser.add_argument("plan", help="path to output plan file")
    return parser.parse_args()


def get_options(config_file):
    with open(config_file, "r") as cfile:
        config_dict = json.load(cfile)

    config = ["--alias", config_dict["alias"]]

    if "queue_type" in config_dict:
        config += ["--grounding-queue", config_dict["queue_type"],
                   "--h2-preprocessor",  # TODO this is not currently set by SMAC
                   ]

        config += ["--incremental-grounding"]
        # TODO add incremental grounding options to config file

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

    # TODO add time + memory limits
    subprocess.run([sys.executable, PLAN_PARTIAL_GROUNDING] +
                   [dk_folder, args.domain, args.problem, "--plan", "args.plan"] +
                   config)

    # TODO run plain LAMA as fallback if not solved, yet


if __name__ == "__main__":
    main()
