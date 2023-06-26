import os
import shutil
import logging

class IncumbentSet:
    def __init__(self, TRAINING_DIR, save_model, num_incumbents=3):
        self.data_incumbents = {}
        self.best_incumbents = []
        self.save_model = save_model
        self.INCUMBENTS_DIR = os.path.join(TRAINING_DIR, 'incumbents')
        if os.path.exists(self.INCUMBENTS_DIR):
            shutil.rmtree(self.INCUMBENTS_DIR)
        os.mkdir(self.INCUMBENTS_DIR)
        self.num_incumbents=num_incumbents

        self.incumbent_id = 1

    def add_and_save(self, incumbent, data):
        incumbent_name = f'incumbent-{self.incumbent_id}'
        self.incumbent_id += 1
        shutil.copytree(incumbent, os.path.join(self.INCUMBENTS_DIR, incumbent_name))
        self.data_incumbents [incumbent_name] = data

        if any([self.is_dominated(incumbent_name, inc) for inc in self.best_incumbents]):
            logging.info(f"New incumbent candidate {incumbent_name} was rejected because it is dominated")
            return

        self.best_incumbents = [inc for inc in self.best_incumbents if not self.is_dominated(inc, incumbent_name)]
        self.best_incumbents.append(incumbent_name)

        self.best_incumbents = sorted(self.best_incumbents, key = self.sort_key)

        logging.info(f"New incumbent is top {1+self.best_incumbents.index(incumbent_name)}")
        for x in self.best_incumbents:
            logging.info ("%s %s", x, self.sort_key(x))

        if incumbent_name in self.best_incumbents[:self.num_incumbents]:
            self.save_model.save([ os.path.join(self.INCUMBENTS_DIR, inc) for inc in self.best_incumbents[:self.num_incumbents]])

    def sort_key(self, x):
        data = self.data_incumbents[x]
        coverage = sum([props['coverage'] for _, props in data.items() if 'coverage' in props])
        planner_time = sum([props['planner_time'] for _, props in data.items() if 'planner_time' in props])
        return (-coverage, planner_time)

    def is_dominated(self, i1, i2): # Returns true if i2 is definitively better than i1
        return False # TODO: This is not implemented so it should always return false
        # for ins, props in self.data_incumbents[i1].items():
        #     return False # TODO: Implement this

        # return True # Didn't find any instance where i1 is better than i2
