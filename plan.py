#! /usr/bin/env python

from __future__ import print_function

import argparse
import os.path
import subprocess
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
CEDALION = os.path.join(ROOT, "cedalion")
TRANSLATE = os.path.join(CEDALION, "src", "translate", "translate.py")
PREPROCESS = os.path.join(CEDALION, "src", "preprocess", "preprocess")
SEARCH = os.path.join(CEDALION, "src", "search", "downward-release")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain_knowledge", help="path to domain knowledge file")
    parser.add_argument("domain", help="path to domain file")
    parser.add_argument("problem", help="path to problem file")
    parser.add_argument("plan", help="path to output plan file")
    return parser.parse_args()


def get_configs(portfolio):
    attributes = {}
    with open(portfolio) as portfolio_file:
        content = portfolio_file.read()
        try:
            exec(content, attributes)
        except Exception:
            sys.exit("The file %s could not be loaded" % portfolio)
    if "configs" not in attributes:
        sys.exit("portfolio %s must define configs" % portfolio)
    return attributes["configs"]


def main():
    args = parse_args()
    configs = get_configs(args.domain_knowledge)
    _, config = configs[0]
    subprocess.check_call([sys.executable, TRANSLATE, args.domain, args.problem])
    with open("output.sas") as f:
        subprocess.check_call([PREPROCESS], stdin=f)
    with open("output") as f:
        subprocess.check_call([SEARCH, "--plan-file", args.plan] + config, stdin=f)


if __name__ == "__main__":
    main()
