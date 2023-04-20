import json
import os
import tarfile


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

# Takes a dictionary from instances to properties, as well as

def select_instances_from_properties(instances_with_properties, conditions):
    return [ins for ins, p in instances_with_properties.items() if all([c(ins, p) for c in conditions])]

def planner_time_under(runtime):
    return lambda i, p : p['coverage']  and p['planner_time'] < 120

def in_instanceset(instance_set):
    return lambda i, p : i in instance_set

def notin_instanceset(instance_set):
    return lambda i, p : i not in instance_set

def not_solved():
    return lambda i, p : p['coverage'] == 0



def num_instances_from_properties(instances_with_properties, f):
    return len(select_instances_from_properties(instances_with_properties, f))

def save_model(source_dir, knowledge_file):
    with tarfile.open(knowledge_file, "w:gz", dereference=True) as tar:
        for f in os.listdir(source_dir):
            tar.add(os.path.join(source_dir, f), arcname=f)
