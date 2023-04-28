import os
import shutil

class IncumbentSet:
    def __init__(self, TRAINING_DIR, save_model, num_incumbents=3):
        self.data_incumbents = {}
        self.best_incumbents = []
        self.save_model = save_model
        if os.path.exists(os.path.join(TRAINING_DIR, 'incumbents')):
            shutil.rmtree(TRAINING_DIR)
        self.INCUMBENTS_DIR = os.path.join(TRAINING_DIR, 'incumbents')

        self.incumbent_id = 1

    def add(self, incumbent, data):
        incumbent_name = f'incumbent-{self.incumbent_id}'
        self.incumbent_id += 1
        shutil.copytree(incumbent, os.path.join(self.INCUMBENTS_DIR, incumbent_name))

        if any([self.is_dominated(data, other_data) for o, other in self.data_incumbents]):
            print(f"New incumbent candidate {incumbent_name} was rejected because it is dominated")
            return

        self.best_incumbents.append(incumbent)
        self.data_incumbents [incumbent_name] = data

        self.best_incumbents = sorted(self.best_incumbents, key = self.sort_key)

        print(f"New incumbent is top {1+self.best_incumbents.index(incumbent_name)}")

        if incumbent_name in self.best_incumbents[:num_incumbents]:
            save_model.save([ os.path.join(self.INCUMBENTS_DIR, inc) for inc in self.best_incumbents[:num_incumbents]])

    def sort_key(self, x): # Returns true if i2 is definitively better than i1
        data = self.data_incumbents[x]
        coverage = sum([props['coverage'] for ins, props in data.items() if 'coverage' in props])
        pass
    def is_dominated(self, i1, i2): # Returns true if i2 is definitively better than i1
        for ins, props in self.data_incumbents[i1].items():
            pass

        return True # Didn't find any instance
