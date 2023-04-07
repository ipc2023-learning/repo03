import os
import sys
from lab.calls.call import Call

def run_step_partial_grounding_rules(REPO_LEARNING, RUNS_DIR, WORKING_DIR, domain_file, time_limit=300, memory_limit = 4*1024*1024):
    #TODO: check time and memory limit (right now it's taken as a limit per step, and not a limit in total

    os.mkdir(f"{WORKING_DIR}")    # TODO: Set to 10k instead of 1k
    Call([sys.executable, f'{REPO_LEARNING}/learning-sklearn/generate-exhaustive-feature-rules.py', domain_file, '--runs', f'{RUNS_DIR}', '--rule_size', '7', '--store_rules', f'{WORKING_DIR}/rules-exhaustive-1k', '--num_rules','1000'], "generate-rules", time_limit=time_limit, memory_limit=memory_limit).wait()
    # TODO: Check if rules have been correctly generated. Otherwise, re-generate with smaller size?

    Call([sys.executable, f'{REPO_LEARNING}/learning-sklearn/filter-irrelevant-rules.py', '--instances-relevant-rules', '10', f'{RUNS_DIR}', f'{WORKING_DIR}/rules-exhaustive-1k', f'{WORKING_DIR}/rules-exhaustive-1k-filtered'], "filter-rules", time_limit=time_limit, memory_limit=memory_limit).wait()

    Call([sys.executable, f'{REPO_LEARNING}/learning-sklearn/generate-training-data.py', \
                                         f'{RUNS_DIR}',\
                                         f'{WORKING_DIR}/rules-exhaustive-1k-filtered',\
                                         f'{WORKING_DIR}/training-data-good-operators-exhaustive-1k-filtered',\
                                         '--op-file', 'good_operators',\
                                         '--max-training-examples', '1000000' # '--num-test-instances TODO Set some test instances
          ], "generate-training-data-1", time_limit=time_limit, memory_limit=memory_limit).wait()



    # TODO: Consider here more feature selection methods, possibly parameterized
    feature_selection_methods = ["DT"]

    for method in feature_selection_methods:
        Call([sys.executable, f'{REPO_LEARNING}/learning-sklearn/feature-selection.py', '--training-folder', f'{WORKING_DIR}/training-data-good-operators-exhaustive-1k-filtered', '--selector-type', method], "feature-selection", time_limit=time_limit, memory_limit=memory_limit).wait()

    # Generate training data for all files of useful rules
    useful_rules_files = [f for f in os.listdir( f'{WORKING_DIR}/training-data-good-operators-exhaustive-1k-filtered') if f.startswith('useful_rules')]
    for useful_rules_file in useful_rules_files:
        Call([sys.executable, f'{REPO_LEARNING}/learning-sklearn/generate-training-data.py', \
              f'{RUNS_DIR}',\
              f'{WORKING_DIR}/training-data-good-operators-exhaustive-1k-filtered/{useful_rules_file}',\
              f'{WORKING_DIR}/training-data-good-operators-exhaustive-1k-{useful_rules_file}',\
              '--op-file', 'good_operators',\
              '--max-training-examples', '1000000' # '--num-test-instances TODO Set some test instances
              ], "generate-training-data", time_limit=time_limit, memory_limit=memory_limit).wait()



    training_data_directories = [f for f in os.listdir( f'{WORKING_DIR}/') if f.startswith('training-data')]

    # TODO: consider here more learning methods, possibly parameterized
    learning_methods = [("DT", ["--model-type", "DT"])]
    for training_data_dir in training_data_directories:
        for learning_method_name, learning_method_parameters in learning_methods:

            output_name = f'model_{training_data_dir.replace("training-data-", "")}_{learning_method_name}'
            os.mkdir (f'{WORKING_DIR}/{output_name}')
            Call([sys.executable, f'{REPO_LEARNING}/learning-sklearn/train-model.py', \
                  '--training-set-folder', f'{WORKING_DIR}/{training_data_dir}',\
                  '--model-folder', f'{WORKING_DIR}/{output_name}',
                  ] + learning_method_parameters, "train", time_limit=time_limit, memory_limit=memory_limit).wait()
