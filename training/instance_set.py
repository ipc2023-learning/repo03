
import json
import os


def select_instances_from_runs (RUNS, f):
    result = []
    for run in os.listdir(RUNS):
        if os.path.join(RUNS, run):
            try:
                with open(os.path.join(RUNS, run, 'properties')) as pfile:
                    content = json.load(pfile)
                    if f (content):
                        result.append(run)
            except:
                pass

    return result

def select_instances_from_runs_with_properties(RUNS, conditions = [], properties=None, only_if_properties_defined=False):
    result = {}
    for run in os.listdir(RUNS):
        if os.path.join(RUNS, run):
            try:
                with open(os.path.join(RUNS, run, 'properties')) as pfile:
                    content = json.load(pfile)
                    if all(c(run, content) for c in conditions):
                        if properties:
                            if only_if_properties_defined and not all([p in content for p in properties]):
                                continue

                            result[run] = {p : content[p] for p in properties if p in content}
                        else:
                            result[run] = content
            except:
                print("Error while retrieving properties from run", run )
                pass

    return result

def planner_time_under(runtime):
    return lambda i, p : p['coverage']  and p['planner_time'] < 120

def in_instanceset(instance_set):
    return lambda i, p : i in instance_set

def notin_instanceset(instance_set):
    return lambda i, p : i not in instance_set

def not_solved(i, p):
    assert 'coverage' in p
    return not p['coverage']

def num_instances_from_properties(instances_with_properties, f):
    return len(select_instances_from_properties(instances_with_properties, f))


class InstanceSet:

    def __init__(self, runs_lama):
        self.instances_with_properties = select_instances_from_runs_with_properties(runs_lama)
        self.training_data_run_dirs = []

        if self.num_instances([not_solved]) == 0:
            print ("Warning: all instances are solved by lama. Make sure your training instances contain difficult instances that are representative of the difficulty of the domain")

        self.INSTANCES_WITH_TRAINING_DATA = set()

    def num_instances (self, conditions):
        return len(self.select_instances(conditions))

    def select_instances (self, conditions):
        result = []
        for ins, p in self.instances_with_properties.items():
            try:
                if all([c(ins, p) for c in conditions]):
                   result.append(ins)
            except:
                pass

        return result

    def add_training_data(self, runs_training_data):
        self.training_data_run_dirs.append(runs_training_data)

    def get_training_datasets(self):
        return self.training_data_run_dirs


        ######
        ### We split instances between training and SMAC optimization following the criteria:
        ###   (1) Any instance not solved by LAMA is for SMAC optimization
        ###   (2) We need to include at least 3 instances that are solved by lama between 10 and 300 seconds
        ###   (3) We want to minimize the overlap with instances for which we have good operators
        ######

    def split_training_instances(self):
        self.INSTANCES_WITH_TRAINING_DATA = set()
        for runs_training_data in self.training_data_run_dirs:
                self.INSTANCES_WITH_TRAINING_DATA.update(select_instances_from_runs(runs_training_data, lambda p : p['coverage']))

        # Here, we have instances reserved for using SMAC (not all of them need to be used). This includes all the instances for which we do not have training data
        self.SMAC_INSTANCES = set(self.select_instances([notin_instanceset(self.INSTANCES_WITH_TRAINING_DATA)]))


        # Make sure that at least 3 instances are solved by lama under 2 minutes
        if self.num_instances ([planner_time_under(120), in_instanceset(self.SMAC_INSTANCES) ]) < 3:
            candidate_instances = self.select_instances ([planner_time_under(120), notin_instanceset(self.SMAC_INSTANCES) ])
            sorted_instances = sorted(candidate_instances, key = lambda x : self.instances_with_properties[x]['planner_time'])

            self.SMAC_INSTANCES.update(sorted_instances[-3:]) # Pick the last three instances

            if len(self.INSTANCES_WITH_TRAINING_DATA) > 10:
                self.INSTANCES_WITH_TRAINING_DATA = [ins for ins in self.INSTANCES_WITH_TRAINING_DATA if ins not in self.SMAC_INSTANCES]


        while (self.num_instances([planner_time_under(120), in_instanceset(self.SMAC_INSTANCES)]) < len(self.INSTANCES_WITH_TRAINING_DATA)/10):
            candidate_instances = self.select_instances ([planner_time_under(120), notin_instanceset(self.SMAC_INSTANCES) ])

            if (candidate_instances):
                sorted_instances = sorted(candidate_instances, key = lambda x : self.instances_with_properties[x]['planner_time'])

                self.SMAC_INSTANCES.add(sorted_instances[-1]) # Pick the last instance

                assert len(self.INSTANCES_WITH_TRAINING_DATA) > 10
                self.INSTANCES_WITH_TRAINING_DATA = [ins for ins in self.INSTANCES_WITH_TRAINING_DATA if ins not in self.SMAC_INSTANCES]

        num_smac_instances_unsolved = self.num_instances([not_solved, in_instanceset(self.SMAC_INSTANCES) ])
        num_smac_instances_under_2m = self.num_instances([planner_time_under(120), in_instanceset(self.SMAC_INSTANCES) ])
        num_smac_instances_overlap = self.num_instances([in_instanceset(self.INSTANCES_WITH_TRAINING_DATA), in_instanceset(self.SMAC_INSTANCES) ])

        print (f"After the split, we have {len(self.INSTANCES_WITH_TRAINING_DATA)} instances for training, {len(self.SMAC_INSTANCES)} for hyperparameter optimization, out of which {num_smac_instances_overlap} have overlap and {num_smac_instances_unsolved} are not solved by lama and {num_smac_instances_under_2m} are solved under 2 minutes)")
        print ("Instances for the training phase: ", self.INSTANCES_WITH_TRAINING_DATA)
        print ("Instances for SMAC optimization: ", self.SMAC_INSTANCES)

        return self.INSTANCES_WITH_TRAINING_DATA


    def get_smac_instances(self, properties):

        selected_instances = []
        for ins in self.SMAC_INSTANCES:
            selected_instances.append(ins)


        max_value = {}
        for p in properties:
            max_value [p] = max([self.instances_with_properties[ins][p] for ins in self.SMAC_INSTANCES if p in self.instances_with_properties[ins]])

        selected_smac_instances = {}
        for ins in self.SMAC_INSTANCES:
            selected_smac_instances [ins] = [self.instances_with_properties[ins][p] if p in self.instances_with_properties[ins] else max_value[p]*10 for p in properties]

        return selected_smac_instances


    # SMAC_INSTANCES_FIRST_OPTIMIZATION =  select_instances_from_runs_with_properties(f'{TRAINING_DIR}/runs-lama', [in_instanceset(self.SMAC_INSTANCES)],
    #                                                                                     ['translator_operators', 'translator_facts', 'translator_variables'])
    # # Make sure that we have 10 instances for SMAC. Preferably, instances that were not solved by good operators.
    # # Score instances according to how far they are from the desired range, pick 10 best (diversifying)
    # assert (len(self.SMAC_INSTANCES_FIRST_OPTIMIZATION))
