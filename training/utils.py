import os
import tarfile
import shutil

import traceback

from collections import defaultdict

from lab.calls.call import Call
import sys

class SaveModel:
    def __init__(self, knowledge_file, keep_copies=True):
        self.knowledge_file = knowledge_file
        self.keep_copies = 1 if keep_copies else 0

    def save(self, source_dirs):
        if not self.knowledge_file:
            return

        assert isinstance(source_dirs, list)

        with tarfile.open(self.knowledge_file + '.tmp', "w:gz", dereference=True) as tar:
            id = 0
            for source_dir in source_dirs:
                for f in os.listdir(source_dir):
                    arcname = os.path.join(os.path.basename(os.path.normpath(source_dir)) + f"_{id}", f)
                    tar.add(os.path.join(source_dir, f), arcname=arcname)
                id += 1

        knowledge_filename = self.knowledge_file
        if self.keep_copies:
            knowledge_filename += f'.{self.keep_copies}'
            self.keep_copies += 1

        shutil.move(self.knowledge_file + '.tmp', knowledge_filename)


def filter_training_set(REPO_LEARNING, TRAINING_SET, rules_file, output):
    # try:
        os.mkdir(output)
        for problem in os.listdir(TRAINING_SET):
            if not os.path.exists(os.path.join(TRAINING_SET, problem, 'good_operators')):
                continue
            output_p = os.path.join(output, problem)
            os.mkdir(output_p)
            shutil.copy(os.path.join(TRAINING_SET, problem, 'domain.pddl'), output_p)
            shutil.copy(os.path.join(TRAINING_SET, problem, 'problem.pddl'), output_p)

            Call([sys.executable, f'{REPO_LEARNING}/learning-sklearn/filter-operators.py',
                  os.path.join(TRAINING_SET, problem, 'domain.pddl'),
                  os.path.join(TRAINING_SET, problem, 'problem.pddl'),
                  os.path.join(TRAINING_SET, problem, 'all_operators'),
                  rules_file,
                  os.path.join(output_p, 'all_operators')
                  ], "filter-training-set").wait()


            Call([sys.executable, f'{REPO_LEARNING}/learning-sklearn/filter-operators.py',
                  os.path.join(TRAINING_SET, problem, 'domain.pddl'),
                  os.path.join(TRAINING_SET, problem, 'problem.pddl'),
                  os.path.join(TRAINING_SET, problem, 'good_operators'),
                  rules_file,
                  os.path.join(output_p, 'good_operators')], "filter-training-set").wait()

    # except Exception as e:
    #     print (f"Warning: Error while filtering training set: {e}")
    #     pass

def combine_training_sets(TRAINING_SETS, output):
    try:
        os.mkdir(output)
        good_operators = defaultdict(set)
        for TRAINING_SET in TRAINING_SETS:
            for problem in os.listdir(TRAINING_SET):
                if not os.path.exists(os.path.join(TRAINING_SET, problem, 'good_operators')):
                    continue
                output_p = os.path.join(output, problem)

                if not os.path.exists(output_p):
                    os.mkdir(output_p)
                    shutil.copy(os.path.join(TRAINING_SET, problem, 'all_operators'), output_p)
                    shutil.copy(os.path.join(TRAINING_SET, problem, 'domain.pddl'), output_p)
                    shutil.copy(os.path.join(TRAINING_SET, problem, 'problem.pddl'), output_p)
                else:
                    pass # TODO assert that files are equal

                with open(os.path.join(TRAINING_SET, problem, 'good_operators')) as f:
                    good_operators[problem].update (f.readlines())

        for problem in good_operators:
            output_p = os.path.join(output, problem)

            with open(os.path.join(output_p, 'good_operators'), 'w') as fw:
                for action in sorted(good_operators[problem]):
                    fw.write(action)

    except Exception as e:
        print (f"Warning: Error while combining training sets: {repr(e)}")
