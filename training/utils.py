import json
import os

def select_instances_by_properties(RUNS, f):
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
