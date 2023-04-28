import os
import sys
from lab.calls.call import Call

def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def run_step_partial_grounding_rules(REPO_LEARNING, RUNS_DIRS, WORKING_DIR, domain_file, time_limit=300,
                                     memory_limit = 4*1024*1024):
    #TODO: check time and memory limit (right now it's taken as a limit per step, and not a limit in total

    os.mkdir(f"{WORKING_DIR}")

    RUNS_DIR = RUNS_DIRS[-1] # The first steps do not depend on the set of good operators so we just use the last directory

    Call([sys.executable, f'{REPO_LEARNING}/learning-sklearn/generate-exhaustive-feature-rules.py', domain_file, '--runs', RUNS_DIR, '--rule_size', '10', '--store_rules', f'{WORKING_DIR}/rules-exhaustive', '--num_rules','10000', '--max_num_rules','20000', '--schema_time_limit', '100'], "generate-rules", time_limit=time_limit, memory_limit=memory_limit).wait()
    # TODO: Check if rules have been correctly generated. Otherwise, re-generate with smaller size?

    Call([sys.executable, f'{REPO_LEARNING}/learning-sklearn/filter-irrelevant-rules.py', '--instances-relevant-rules', '10',
          f'{RUNS_DIR}', f'{WORKING_DIR}/rules-exhaustive', f'{WORKING_DIR}/rules-exhaustive-filtered',
          '--time-limit', str(time_limit)], "filter-rules", time_limit=time_limit*10, memory_limit=memory_limit).wait()


    # Check if rules have been correctly generated. Otherwise, re-generate with smaller size
    if not is_non_zero_file(f'{WORKING_DIR}/rules-exhaustive-filtered'):
        Call([sys.executable, f'{REPO_LEARNING}/learning-sklearn/generate-exhaustive-feature-rules.py', domain_file, '--runs', RUNS_DIR, '--rule_size', '5', '--store_rules', f'{WORKING_DIR}/rules-exhaustive', '--num_rules','1000', '--max_num_rules','2000', '--schema_time_limit', '100'], "generate-rules", time_limit=time_limit, memory_limit=memory_limit).wait()

        Call([sys.executable, f'{REPO_LEARNING}/learning-sklearn/filter-irrelevant-rules.py', '--instances-relevant-rules', '10', f'{RUNS_DIR}', f'{WORKING_DIR}/rules-exhaustive', f'{WORKING_DIR}/rules-exhaustive-filtered', '--time-limit', str(time_limit)], "filter-rules", time_limit=time_limit*10, memory_limit=memory_limit).wait()


    if not is_non_zero_file(f'{WORKING_DIR}/rules-exhaustive-filtered'):
        Call([sys.executable, f'{REPO_LEARNING}/learning-sklearn/generate-exhaustive-feature-rules.py', domain_file, '--runs', RUNS_DIR, '--rule_size', '5', '--store_rules', f'{WORKING_DIR}/rules-exhaustive', '--num_rules','100', '--max_num_rules','200', '--schema_time_limit', '100'], "generate-rules", time_limit=time_limit, memory_limit=memory_limit).wait()

        Call([sys.executable, f'{REPO_LEARNING}/learning-sklearn/filter-irrelevant-rules.py', '--instances-relevant-rules', '10', f'{RUNS_DIR}', f'{WORKING_DIR}/rules-exhaustive', f'{WORKING_DIR}/rules-exhaustive-filtered', '--time-limit', str(time_limit)], "filter-rules", time_limit=time_limit*10, memory_limit=memory_limit).wait()


    if not is_non_zero_file(f'{WORKING_DIR}/rules-exhaustive-filtered'):
        return

    # This step depends on the set of good operators, but performing it for all data-sets gives too many combinations
    Call([sys.executable, f'{REPO_LEARNING}/learning-sklearn/generate-training-data.py',
                                         f'{RUNS_DIR}',
                                         f'{WORKING_DIR}/rules-exhaustive-filtered',
                                         f'{WORKING_DIR}/training-data-good-operators-exhaustive-filtered',
                                         '--op-file', 'good_operators',
                                         '--max-training-examples', '1000000', '--time-limit', str(time_limit) # '--num-test-instances TODO Set some test instances
          ], "generate-training-data-1", time_limit=time_limit*10, memory_limit=memory_limit).wait()


    # One could consider here more feature selection methods, possibly parameterized
    # However, not sure if it's worth playing around with this; in some local tests, it looks like
    # DT gives by far the best scores. The other models mostly give similar scores to very many
    # rules, so it's not clear if these scores say a lot about the relevance.
    # feature_selection_methods = ["DT", "RF", "LINR", "LOGR", "SVR"]
    feature_selection_methods = ["DT","LOGR", "SVR"]

    for method in feature_selection_methods:
        Call([sys.executable, f'{REPO_LEARNING}/learning-sklearn/feature-selection.py', '--training-folder', f'{WORKING_DIR}/training-data-good-operators-exhaustive-filtered', '--selector-type', method], "feature-selection", time_limit=time_limit, memory_limit=memory_limit).wait()

    # Generate training data for all files of useful rules
    useful_rules_files = [f for f in os.listdir(f'{WORKING_DIR}/training-data-good-operators-exhaustive-filtered') if f.startswith('useful_rules')]
    for useful_rules_file in useful_rules_files:
        for RUNS_DIR in RUNS_DIRS:
            for op_file in ['sas_plan', 'good_operators']:
                if RUNS_DIR.endswith('combined') and op_file == 'sas_plan':
                    continue # Skip this combination which does not make sense

                Call([sys.executable, f'{REPO_LEARNING}/learning-sklearn/generate-training-data.py', \
                      f'{RUNS_DIR}',\
                      f'{WORKING_DIR}/training-data-good-operators-exhaustive-filtered/{useful_rules_file}',\
                      f'{WORKING_DIR}/training-data-{op_file}-exhaustive-{useful_rules_file}-{os.path.basename(RUNS_DIR)}',\
                      '--op-file', op_file,\
                      '--max-training-examples', '1000000'
                      ], "generate-training-data", time_limit=time_limit, memory_limit=memory_limit).wait()


    training_data_directories = [f for f in os.listdir( f'{WORKING_DIR}/') if f.startswith('training-data')]

    # TODO: consider here more learning methods, possibly parameterized
    learning_methods = [("DT", ["--model-type", "DT"]),
                        ('LOGR', ["--model-type", "LOGR"]),
                        ('LINR', ["--model-type", "LINR"]),
                        ('RF', ["--model-type", "RF"]),
                        ('SVR', ["--model-type", "SVR"]),
                        ('KRN_RG', ["--model-type", "KRN_RG"]),
                        ]
    for training_data_dir in training_data_directories:
        for learning_method_name, learning_method_parameters in learning_methods:

            output_name = f'model_{training_data_dir.replace("training-data-", "")}_{learning_method_name}'
            os.mkdir (f'{WORKING_DIR}/{output_name}')
            Call([sys.executable, f'{REPO_LEARNING}/learning-sklearn/train-model.py', \
                  '--training-set-folder', f'{WORKING_DIR}/{training_data_dir}',\
                  '--model-folder', f'{WORKING_DIR}/{output_name}',
                  ] + learning_method_parameters, "train", time_limit=time_limit, memory_limit=memory_limit).wait()
