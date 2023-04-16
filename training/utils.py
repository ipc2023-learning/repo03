import json
import os

def select_instances (RUNS, f):
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


def select_instances_with_properties(RUNS, f, properties):
    result = {}
    for run in os.listdir(RUNS):
        if os.path.join(RUNS, run):
            try:
                with open(os.path.join(RUNS, run, 'properties')) as pfile:
                    content = json.load(pfile)
                    if f (content) and all([p in content for p in properties]):
                        result[run] = [content[p] in content for p in properties]
            except:
                pass

    return result
