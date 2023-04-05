#! /usr/bin/env python

from __future__ import print_function

import argparse
import os
import os.path
import shutil
import sys

from lab.calls.call import Call
from lab.environments import  LocalEnvironment

sys.path.append(f'{os.path.dirname(__file__)}/training')
from training import TrainingExperiment



def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain", help="path to domain file")
    parser.add_argument("problem", nargs="+", help="path to problem file")

    parser.add_argument("--path", default='./data', help="path to domain file")

    parser.add_argument("--cpus", type=int, default=1, help="number of cpus available")

    parser.add_argument("--total_time_limit", default=30, type=int, help="time limit")
    parser.add_argument("--total_memory_limit", default=7*1024, help="memory limit")

    return parser.parse_args()


def main():
    args = parse_args()

    ROOT = os.path.dirname(os.path.abspath(__file__))

    TRAINING_DIR=args.path

    REPO_GOOD_OPERATORS = f"{ROOT}/fd-symbolic"
    REPO_LEARNING = f"{ROOT}/learning"
    BENCHMARKS_DIR = f"{TRAINING_DIR}/instances"
    REPO_PARTIAL_GROUNDING = f"{ROOT}/fd-partial-grounding"

    if os.path.exists(TRAINING_DIR):
        shutil.rmtree(TRAINING_DIR)
    os.mkdir(TRAINING_DIR)

    # Copy all input benchmarks to the directory
    os.mkdir(BENCHMARKS_DIR)
    shutil.copy(args.domain, BENCHMARKS_DIR)
    for problem in args.problem:
        shutil.copy(problem, BENCHMARKS_DIR)

    ENV = LocalEnvironment(processes=args.cpus)

    exp = TrainingExperiment(TRAINING_DIR, environment=ENV, instances = ["instances"])
    exp.add_default_steps(REPO_GOOD_OPERATORS, REPO_PARTIAL_GROUNDING)

    exp.execute_all_steps()

    training_set = []


        # print("Preprocessing", problem)
        # time_limit = 24 * 60 * 60 / len(args.problem)
        # memory_limit = 7 * 1024
        # Call([sys.executable, TRANSLATE, args.domain, problem], time_limit=time_limit, mem_limit=memory_limit).wait()
        # if not os.path.exists("output.sas"):
        #     continue
        # Call([PREPROCESS], stdin="output.sas", time_limit=time_limit, mem_limit=memory_limit).wait()
        # if not os.path.exists("output"):
        #     continue

        # os.remove("output.sas")
        # problem_path = os.path.join(TRAINING_TASKS_DIR, "p{i:03d}.sas".format(i=len(training_set) + 1))
        # shutil.move("output", problem_path)
        # training_set.append(problem_path)

    # print("Training set:", training_set)
    # with open("instances.txt", "w") as f:
    #     f.write("\n".join(training_set))


if __name__ == "__main__":
    main()
