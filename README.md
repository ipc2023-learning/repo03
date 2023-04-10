# Good Old-Fashioned AI

This repository contains the learning scripts for learning partial grounding and pruning rules models.


## Installation

The dependency yap must be installed locally. To do so, follow the following instructions.

cd yap
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=<installation_directory>
make
make install


## Usage

### Training a model with sklearn
The usage of learning algorithms 2. is divided in two phases:
The training phase first does offline learning of models, the planning phase then uses these learned models during grounding.
For the learning phase there are two alternatives: using **relational rules** with standard learning algorithms, or **inductive logic programming (ILP)** using Aleph.
In what follows, we describe the former option in detail. Aleph can be used very similarly, as indicated below.

**Learning Phase:**
* Prerequisites:
   - python3, with libraries: numpy, sklearn, matplotlib, pandas, pylab

   - A training dataset (the one from Gnad et. al. (2019) can be found here: <https://gitlab.com/atorralba/useful-actions-dataset>) containing:
      - `<domain>` : A PDDL domain file that is shared accross all training and testing instances.
      - `<runs>` : A directory containing the training data. It should contain a sub-directory for each instance, which should contain the following files:
        * domain.pddl
        * problem.pddl
        * sas_plan/good_operators: file containing the list of "good" operators, one per line.
        * all_operators.bz2: file containing a list of all grounded (good and bad) operators, one per line. Compressed with bz2 format to use less space.

The learning phase consists of several steps. They require executing python scripts located in *src/subdominization-training*.

**1. `./generate-exhaustive-feature-rules.py`**: Generate an initial set of features (each feature correspond to a rule).
  It exhaustively generates many rules, and one can control the size by two parameters: (rule_size y num_rules).
  If the training runs are provided, it'll extract data from them to avoid rules that check predicates in the initial state or goal if they never appeared there. This is recommended to avoid unnecessary rules that would be entirely uninformative.

  Usage:
  `./generate-exhaustive-feature-rules.py --store_rules <output_rule_file> --rule_size RULE_SIZE --num_rules NUM_RULES --runs <runs> <domain>`

  Recommended values for `RULE_SIZE` is 10, so that the number of features is controlled by `NUM_RULES`: Higher-values (100k) will require much longer training times than lower values (1K), but also can provide more accuracy in the end.


**1.5 `./filter-irrelevant-rules.py`** (optional): Remove irrelevant rules.
  There is an extra step that can be executed between steps 1 and 2, which is not strictly necessary.
  After step 1, one can filter out features (rules) that have exactly the same value in all the cases in the training data (these rules are simply invariants, so they are not useful features for the learning algorithms), or complex rules if there exists a shorter rule that is equivalent in the training set (they always evaluate to the same value).

  Usage:
  `./filter-irrelevant-rules.py  [--instances-relevant-rules INSTANCES_RELEVANT_RULES] [--max-training-examples MAX_TRAINING_EXAMPLES] <runs> <training_rules> <output>`
  `<training_rules>` is the file generated in step 1.
  The two parameters are optional and make the rule filter approximate in exchange of a faster check, and to filter features that can be relevant but only in very few training examples.


**2. `./generate-training-data.py`**: Generate the training data.

  Usage:
  `generate-training-data.py [--debug-info] [--instances-relevant-rules INSTANCES_RELEVANT_RULES] [--op-file OP_FILE] [--num-test-instances NUM_TEST_INSTANCES] [--max-training-examples MAX_TRAINING_EXAMPLES] <runs_folder> <training_rules> <output_path_to_store_training_data>`

   - `NUM_TEST_INSTANCES` allows you to separate some instances to validate the model.

   - `OP_FILE` allows you to control the name of the file that you want to use as "good_operators". By default is sas_plan, but you can have different files with different operator subsets and use this to control which ones to use for training the models.

   - `INSTANCES_RELEVANT_RULES` and `MAX_TRAINING_EXAMPLES` allow you to filter the input features that are irrelevant (see step 1-prime below)


**3. `./learning/select-features.py`** (optional but highly recommended): Feature selection (subset of rules).

  Usage:
  `./learning/select-features.py --training-folder FOLDER1 --selector-type TYPE [--keep-duplicate-features] [--mean-over-duplicates]`

  - --training-folder: path to training set files (must be *.csv, where last column is the class, also need relevant_rules file); this is the outcome of 2)
  - --selector-type: the type of the learning model: can be one of 'LRCV', 'LG', 'RF' , 'SVMCV','NBB', 'NBG', 'DT'
  - --keep-duplicate-features: elimination and aggregation of duplicate feature vectors, default is eliminate
  - --mean-over-duplicates: aggregating eliminated duplicate feature vectors by taking max or mean (default is max)

**4. `./generate-training-data.py`** (only needed if 3. was done): Re-generate the training data with the reduced rule set of 3.

As `<training_rules>`, the `useful_rules_X` file generated in 3. has to be specified, where X depends on the choice of the selector (we always used DT).

**5. `../train-model.py`**: Train the model.

  Usage:
  `./train-model.py --training-set-folder FOLDER1 --model-folder FOLDER2 --model-type TYPE [--keep-duplicate-features] [--mean-over-duplicates]`

  - --training-set-folder:  path to training set files (must be *.csv, where last column is the class); this is the outcome of 2) or 3)
  - --model-folder: path to folder where to store model files in
  - --model-type: the type of the learning model: can be one of 'LRCV', 'LG', 'RF' , 'SVMCV','NBB', 'NBG', 'DT'
  - --keep-duplicate-features: elimination and aggregation of duplicate feature vectors, default is eliminate
  - --mean-over-duplicates: when --keep-duplicate-features is set, aggregating eliminated duplicate feature vectors by taking max or mean (default is max)


The result of step 5. is a folder containing the *models* as well as the *relevant_rules* that can be loaded into our version of FastDownward (see above)



# Training a model with Aleph

1. `generate-training-data-aleph.py`

2. run aleph scripts that are generated by the previous step and redirect the output to a file.

3. run `parse-aleph-theory.py` on the output to generate the Aleph-based models that can be loaded into our version of Fast Downward.


# Dependencies

The following dependencies are included.

## Yap

Taken from: https://github.com/vscosta/yap

Yap is distributed under the   LGPL  licence terms. For details visit http://www.gnu.org/copyleft/lesser.html.

## Aleph

Taken from: https://www.cs.ox.ac.uk/activities/programinduction/Aleph/aleph.html

## Translate

Taken from:  https://www.fast-downward.org/  (release 22-12)

Fast Downward is licensed under the GNU Public License (GPL), as described in the main repository. If you want to use the planner in any way that is not compatible with the GPL (v3 or newer), you will have to get permission from us.
