#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os.path

from .priority_queue import SortedHeapQueue, FIFOQueue
from .queue_factory import PriorityQueue
from .model import HybridModel

from collections import defaultdict
import itertools
from math import inf


# TODO add support for writing/reading action list from file
# TODO add support for good/bad action rules

def get_hybrid_model(task, args):
    if not args.trained_model_folder:
        exit("ERROR: need to provide --trained-model-folder")
    return HybridModel(args.trained_model_folder, task)


class IPC23SingleQueue(PriorityQueue):
    def __init__(self, task, args):
        self.queue = SortedHeapQueue(False)
        self.closed = []
        self.model = get_hybrid_model(task, args)
        self.batch_eval = args.batch_evaluation
        self.non_evaluated_actions = defaultdict(list)

    def __bool__(self):
        return bool(self.queue) or \
            (self.batch_eval and any(bool(actions) for actions in self.non_evaluated_actions.values()))
    __nonzero__ = __bool__

    def get_final_queue(self):
        return self.closed

    def print_info(self):
        print("Using IPC23 single-queue with a hybrid trained model (rules+aleph) for actions.")
        if self.batch_eval:
            print("Actions are evaluated in batches.")

    def print_stats(self):
        self.model.print_stats()

    def get_hard_action_if_exists(self, is_hard_action):
        raise

    def notify_new_hard_actions(self, schemas):
        raise

    def push(self, action):
        if self.batch_eval:
            self.non_evaluated_actions[action.predicate.name].append(action)
        else:
            estimate = self.model.get_estimate(action)
            if estimate is None:
                # give high probability if we don't have a model
                estimate = 1.0
            self.queue.push(action, estimate)

    def pop(self):
        if self.batch_eval:
            for schema, actions in self.non_evaluated_actions.items():
                if actions:
                    estimates = self.model.get_estimates(actions)
                    if not estimates:
                        # give high probability if we don't have a model
                        estimates = [1.0 for i in range(len(actions))]
                    self.queue.push_list(actions, estimates)
                    self.non_evaluated_actions[schema] = []
        action = self.queue.pop()
        self.closed.append(action)
        return action


class IPC23RoundRobinQueue(PriorityQueue):
    def __init__(self, task, args):
        self.model = get_hybrid_model(task, args)
        self.schemas = list(self.model.get_trained_schemas())
        self.num_trained_schemas = len(self.schemas)
        self.current = 0
        self.queues = [SortedHeapQueue(False) for _ in self.schemas]
        self.num_grounded_actions = [0 for _ in self.schemas]
        self.closed = []
        self.batch_eval = args.batch_evaluation
        if self.batch_eval:
            self.non_evaluated_actions = [[] for _ in self.schemas]
        self.has_actions = False

    def __bool__(self):
        return self.has_actions
    __nonzero__ = __bool__

    def get_final_queue(self):
        return self.closed + list(itertools.chain.from_iterable(
            self.queues[i].get_final_queue() for i in range(self.num_trained_schemas, len(self.queues))))

    def print_info(self):
        print("Using IPC23 round-robin queue with a hybrid trained model (rules+aleph) for actions.")
        if self.batch_eval:
            print("Actions are evaluated in batches.")

    def print_stats(self):
        self.model.print_stats()
        for i in range(len(self.num_grounded_actions)):
            print(f"{self.num_grounded_actions[i]} actions grounded for schema {self.schemas[i]}")

    def get_hard_action_if_exists(self, is_hard_action):
        pass

    def notify_new_hard_actions(self):
        pass

    def push(self, action):
        self.has_actions = True
        schema = action.predicate.name
        if schema not in self.schemas:
            # first time we see this action schema; init datastructures
            # TODO think of a more efficient way to implement this;
            # can we obtain all schemas from the task initially?
            self.schemas.append(schema)
            self.num_grounded_actions.append(0)
            self.queues.append(FIFOQueue())
            self.queues[-1].push(action)
        else:
            index = self.schemas.index(schema)
            if index < self.num_trained_schemas:
                # we have a model
                if self.batch_eval:
                    self.non_evaluated_actions[index].append(action)
                else:
                    estimate = self.model.get_estimate(action)
                    assert estimate is not None
                    self.queues[index].push(action, estimate)
            else:
                # we don't have a trained model
                self.queues[index].push(action)

    def _evaluate_batch_schema(self, index):
        actions = self.non_evaluated_actions[index]
        if actions:
            estimates = self.model.get_estimates(actions)
            self.queues[index].push_list(actions, estimates)
            self.non_evaluated_actions[index] = []

    def _check_has_action(self):
        self.has_actions = any(q for q in self.queues) or \
                           (self.batch_eval and any(q for q in self.non_evaluated_actions))

    def pop(self):
        self.has_actions = False
        while True:
            self.current = (self.current + 1) % len(self.schemas)
            if self.batch_eval and self.current < self.num_trained_schemas:
                self._evaluate_batch_schema(self.current)
            if self.queues[self.current]:
                self.num_grounded_actions[self.current] += 1
                action = self.queues[self.current].pop()
                if self.queues[self.current]:
                    self.has_actions = True
                else:
                    self._check_has_action()
                if self.current < self.num_trained_schemas:
                    # Only add to closed if there is no model for this schema,
                    # as these actions are kept in the corresponding FIFOQueue.
                    self.closed.append(action)
                return action


class IPC23RatioQueue(PriorityQueue):
    def __init__(self, task, args):
        self.model = get_hybrid_model(task, args)
        self.trained_schemas = set(self.model.get_trained_schemas())

        ratios_file_name = os.path.join(args.trained_model_folder, "schema_ratios")
        if not os.path.isfile(ratios_file_name):
            exit("ERROR: no schema_ratios file in trained model folder.")
        with open(ratios_file_name, "r") as ratios:
            self.target_ratios = []
            self.ratios = []
            self.schemas = []
            self.queues = []
            self.num_grounded_actions = []
            self.num_actions = []
            for line in ratios:
                schema, ratio = line.split(":")
                self.target_ratios.append(float(ratio.strip()))
                self.ratios.append(0.0)
                self.schemas.append(schema.strip())
                if schema in self.trained_schemas:
                    self.queues.append(SortedHeapQueue(False))
                else:
                    self.queues.append(FIFOQueue())
                self.num_grounded_actions.append(0)

        self.closed = []
        self.skipped_action_schemas = set()

        self.batch_eval = args.batch_evaluation
        if self.batch_eval:
            self.non_evaluated_actions = [[] for _ in self.schemas]

    def __bool__(self):
        return any(q for q in self.queues) or (self.batch_eval and any(q for q in self.non_evaluated_actions))
    __nonzero__ = __bool__

    def get_final_queue(self):
        return self.closed + list(itertools.chain.from_iterable(
            q.get_final_queue() for i, q in enumerate(self.queues) if self.schemas[i] not in self.trained_schemas))

    def print_info(self):
        print("Using IPC23 schema-ratio queue with a hybrid trained model (rules+aleph) for actions.")
        if self.batch_eval:
            print("Actions are evaluated in batches.")

    def print_stats(self):
        self.model.print_stats()
        for i in range(len(self.num_grounded_actions)):
            print(f"{self.num_grounded_actions[i]} actions grounded for schema {self.schemas[i]}; "
                  f"target ratio: {self.target_ratios[i]}, final ratio: {self.ratios[i]}")
        for schema in self.skipped_action_schemas:
            print(f"WARNING: Action schema {schema} did not appear in given ratios file, so was pruned completely.")

    def get_hard_action_if_exists(self, is_hard_action):
        pass

    def notify_new_hard_actions(self):
        pass

    def push(self, action):
        schema = action.predicate.name
        if schema not in self.schemas:
            self.skipped_action_schemas.add(schema)
            return
        index = self.schemas.index(schema)
        if schema not in self.trained_schemas:
            self.queues[index].push(action)
        else:
            if self.batch_eval:
                self.non_evaluated_actions[index].append(action)
            else:
                self.queues[index].push(action, self.model.get_estimate(action))

    def pop(self):
        next_index = max([(self.target_ratios[i] - self.ratios[i], i)
                          if self.queues[i] or (self.batch_eval and self.non_evaluated_actions[i]) else (-inf, i)
                          for i in range(len(self.ratios))], key=lambda item: item[0])[1]

        if self.batch_eval and self.non_evaluated_actions[next_index]:
            actions = self.non_evaluated_actions[next_index]
            self.queues[next_index].push_list(actions, self.model.get_estimates(actions))
            self.non_evaluated_actions[next_index] = []

        action = self.queues[next_index].pop()
        if action.predicate.name in self.trained_schemas:
            # Only add to closed if there is no model for this schema,
            # as these actions are kept in the corresponding FIFOQueue.
            self.closed.append(action)

        self.num_grounded_actions[next_index] += 1
        total_num_grounded = sum(self.num_grounded_actions)
        for i in range(len(self.ratios)):
            self.ratios[i] = self.num_grounded_actions[i] / total_num_grounded

        return action
