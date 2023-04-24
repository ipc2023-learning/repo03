#! /usr/bin/env python
# -*- coding: utf-8 -*-

import options
from .queue_factory import PriorityQueue

from collections import defaultdict
import heapq
import itertools
from math import inf
import sys

import random

        
class SortedHeapQueue:
    def __init__(self, min_wins=True):
        self.queue = []
        self.count = 0  # this speeds up the queue significantly
        self.min_wins = min_wins  # if true, return minimal element, if false return maximal

    def __bool__(self):
        return bool(self.queue)
    __nonzero__ = __bool__

    def get_hard_action_if_exists(self, is_hard_action):
#         return [action for estimate, action in self.queue if is_hard_action(action)] 
# did not help too much + can even have negative impact on runtime
# probably try this as a parameter n; where each call returns up to n hard actions
        for i in range(len(self.queue)):
            action = self.queue[i][1]
            if is_hard_action(action):
                del self.queue[i]
                heapq.heapify(self.queue)
                return [action]
        return []

    def notify_new_hard_actions(self):
        pass

    def push(self, action, estimate):
        if self.min_wins:
            heapq.heappush(self.queue, (estimate, self.count, action))
        else:
            heapq.heappush(self.queue, (-estimate, self.count, action))
        self.count += 1

    def push_list(self, actions, estimates):
        assert(len(actions) == len(estimates))
        # TODO can this be done more efficient when making actions a heapq itself 
        # and then merging the two?
        for i in range(len(actions)):
            if self.min_wins:
                heapq.heappush(self.queue, (estimates[i], self.count, actions[i]))
            else:
                heapq.heappush(self.queue, (-estimates[i], self.count, actions[i]))
            self.count += 1

    def pop(self):
        return heapq.heappop(self.queue)[2]

    def pop_entry(self):
        entry = heapq.heappop(self.queue)
        if self.min_wins:
            return entry[0], entry[2]
        else:
            return -entry[0], entry[2]


class RandomEvaluator():
    def __init__(self) -> None:
        random_seed = 2023
        if options.random_seed:
            random_seed = options.random_seed
        random.seed(random_seed)
    def get_estimate(self, _):
        return random.randint(0, 10)
    def print_stats(self):
        pass
     
class NoveltyEvaluator():
    def __init__(self):
        self.novelty = {}
    def calculate_novelty(self, action):
        if (not action.predicate.name in self.novelty):
            return 0
        else:
            novelty = sys.maxsize if len(action.args) > 0 else 0 # action without arguments are always novel; no blow-up here
            for i in range(len(action.args)):
                if (action.args[i] in self.novelty[action.predicate.name][i]):
                    novelty = min(novelty, self.novelty[action.predicate.name][i][action.args[i]])
                else:
                    return 0
            return novelty
    def update_novelty(self, action):
        if (not action.predicate.name in self.novelty):
            self.novelty[action.predicate.name] = [{} for i in range(len(action.args))]
            for i in range(len(action.args)):
                self.novelty[action.predicate.name][i][action.args[i]] = 1
        else:
            for i in range(len(action.args)):
                if (action.args[i] in self.novelty[action.predicate.name][i]):
                    self.novelty[action.predicate.name][i][action.args[i]] += 1
                else:
                    self.novelty[action.predicate.name][i][action.args[i]] = 1
    
class FIFOQueue(PriorityQueue):
    def __init__(self):
        self.queue = []
        self.queue_pos = 0
        self.last_hard_index = 0
        self.hard_actions = []
    def __bool__(self):
        return self.queue_pos < len(self.queue)
    __nonzero__ = __bool__
    def get_final_queue(self):
        return self.queue[:self.queue_pos] + self.hard_actions
    def print_info(self):
        print("Using FIFO priority queue for actions.")
    def get_num_grounded_actions(self):
        return self.queue_pos + len(self.hard_actions)
    def get_num_actions(self):
        return len(self.queue) + len(self.hard_actions)
    def get_hard_action_if_exists(self, is_hard_action):
        if (self.last_hard_index < self.queue_pos):
            self.last_hard_index = self.queue_pos
        for i in range(self.last_hard_index, len(self.queue)):
            action = self.queue[i]
            if (is_hard_action(action)):
                self.hard_actions.append(action)
                del self.queue[i]
                return [action]
            self.last_hard_index += 1
        return []
    def notify_new_hard_actions(self, schemas):
        self.last_hard_index = self.queue_pos
    def push(self, action):
        self.queue.append(action)
    def pop(self):
        result = self.queue[self.queue_pos]
        self.queue_pos += 1
        return result
    
class LIFOQueue(PriorityQueue):
    def __init__(self):
        self.queue = []
        self.closed = []
        self.last_hard_index = 0
    def __bool__(self):
        return len(self.queue) > 0
    __nonzero__ = __bool__
    def get_final_queue(self):
        return self.closed
    def print_info(self):
        print("Using LIFO priority queue for actions.")
    def get_num_grounded_actions(self):
        return len(self.closed)
    def get_num_actions(self):
        return len(self.queue) + len(self.closed)
    def get_hard_action_if_exists(self, is_hard_action):
        for i in range(self.last_hard_index, len(self.queue)):
            action = self.queue[i]
            if (is_hard_action(action)):
                self.closed.append(action)
                del self.queue[i]
                return [action]
            self.last_hard_index += 1
        return []
    def notify_new_hard_actions(self, schemas):
        self.last_hard_index = 0
    def push(self, action):
        self.queue.append(action)
    def pop(self):
        result = self.queue.pop()
        if (self.last_hard_index >= len(self.queue)):
            self.last_hard_index = max(0, len(self.queue) - 1)
        self.closed.append(result) 
        return result
    
class ActionsFromFileQueue(PriorityQueue):
    def __init__(self):
        if (not options.actions_file):
            sys.error("need to provide --actionsfile FILE for ActionsFromFileQueue")
        self.actions = defaultdict(set)
        with open(options.actions_file[0], 'r') as afile:
            for action in afile:
                if (action.startswith(";")):
                    # this is probably the "; cost = X (unit cost)" part of a plan file
                    continue
                action = action.replace("(", "").replace(")", "").strip()
                schema, args = action.split(" ", 1)
                schema.strip()
                args.strip()
                args = "".join([x.strip() for x in args.split(" ")])
                self.actions[schema].add(args)
        self.queue = []
        self.not_in_file = []
        self.closed = []
    def __bool__(self):
        while (self.queue):
            result = self.queue[-1]
            schema = result.predicate.name
            args = "".join(result.args)
            if (args in self.actions[schema]):
                return True
            self.not_in_file.append(result)
            self.queue.pop()
        if (options.reachable_actions_output_file): # at this point the grounding is done; somewhat ugly, though
            print("Writing all actions that became reachable during grounding to ", options.reachable_actions_output_file[0])
            with open(options.reachable_actions_output_file[0], 'w') as outfile:
                for action in self.closed + self.not_in_file:
                    outfile.write("{} {}\n".format(action.predicate.name, " ".join(action.args)))
        return False
    __nonzero__ = __bool__
    def get_final_queue(self):
        return self.closed
    def print_info(self):
        print("Using ActionsFromFile priority queue for actions.")
    def get_num_grounded_actions(self):
        return len(self.closed)
    def get_num_actions(self):
        return len(self.queue) + len(self.closed)
    def get_hard_action_if_exists(self, is_hard_action):
        raise "not supported by ActionsFromFileQueue"
    def notify_new_hard_actions(self, schemas):
        raise "not supported by ActionsFromFileQueue"
    def push(self, action):
        self.queue.append(action)
    def pop(self):
        while (self):
            result = self.queue.pop()            
            schema = result.predicate.name
            args = "".join(result.args)
            if (args in self.actions[schema]):
                self.closed.append(result)
                return result
            self.not_in_file.append(result)

class EvaluatorQueue(PriorityQueue):
    def __init__(self, evaluator, info):
        self.queue = SortedHeapQueue(False)
        self.closed = []
        self.model = evaluator
        self.info = info
        self.batch_eval = options.batch_evaluation
        self.non_evaluated_actions = defaultdict(list)
    def __bool__(self):
        return bool(self.queue) or (self.batch_eval and any(bool(actions) for actions in self.non_evaluated_actions.values()))
    __nonzero__ = __bool__
    def get_final_queue(self):
        return self.closed
    def print_info(self):
        print("Using heap priority queue with a trained model for actions.")
        if (self.batch_eval):
            print("Actions are evaluated in batches.")
        print(self.info)
    def print_stats(self):
        self.model.print_stats()
    def get_hard_action_if_exists(self, is_hard_action):
        # TODO support for batch evaluation
        actions = self.queue.get_hard_action_if_exists(is_hard_action)
        if (actions):
            self.closed += actions
        return actions
    def notify_new_hard_actions(self, schemas):
        self.queue.notify_new_hard_actions(schemas)
    def push(self, action):
        if (self.batch_eval):
            self.non_evaluated_actions[action.predicate.name].append(action)
        else:
            estimate = self.model.get_estimate(action)
            if (estimate == None):
                estimate = 1.0
            self.queue.push(action, estimate)
    def pop(self):
        if (self.batch_eval and any(bool(l) for l in self.non_evaluated_actions.values())):
            for schema, actions in self.non_evaluated_actions.items():
                if (actions):
                    estimates = self.model.get_estimates(actions)
                    if not estimates:
                        # we have no model for this action schema
                        estimates = [1.0 for i in range(len(actions))]
                    self.queue.push_list(actions, estimates)
                    self.non_evaluated_actions[schema] = []
        action = self.queue.pop()
        self.closed.append(action)
        return action
    
class RoundRobinQueue(PriorityQueue):
    def __init__(self):
        self.schemas = []
        self.current = 0
        self.queues = []
        self.num_grounded_actions = []
    def __bool__(self):
        for queue in self.queues:
            if (queue):
                return True
        return False
    __nonzero__ = __bool__
    def get_final_queue(self):
        result = []
        for queue in self.queues:
            result += queue.get_final_queue()
        return result
    def print_info(self):
        print("Using SchemaRoundRobin priority queue for actions.")
    def print_stats(self):
        for i in range(len(self.num_grounded_actions)):
            print("%d actions grounded for schema %s" % (self.num_grounded_actions[i], self.schemas[i]))
    def get_hard_action_if_exists(self, is_hard_action):
        for i in range(len(self.queues)):
            action = self.queues[i].get_hard_action_if_exists(is_hard_action)
            if (action):
                self.num_grounded_actions[i] += 1
                return [action]
        return []
    def notify_new_hard_actions(self, schemas):
        for schema in schemas:
            if (schema in self.schemas):
                self.queues[self.queues.index(schema)].notify_new_hard_actions(schemas)
    def push(self, action):
        if (not action.predicate.name in self.schemas):
            self.schemas.append(action.predicate.name)
            self.queues.append(FIFOQueue())
            self.num_grounded_actions.append(0)
        self.queues[self.schemas.index(action.predicate.name)].push(action)
    def pop(self):
        while True:
            self.current = (self.current + 1) % len(self.schemas)
            if (self.queues[self.current]):
                self.num_grounded_actions[self.current] += 1
                return self.queues[self.current].pop()
            
class RatioQueue(PriorityQueue):
    def __init__(self):
        if not options.action_schema_ratios:
            sys.exit("Error: need action schema ratios to use this queue. Please specify using --action-schema-ratios")
        with open(options.action_schema_ratios, "r") as ratios:
            self.target_ratios = []
            self.ratios = []
            self.schemas = []
            self.queues = []
            for line in ratios:
                schema, ratio = line.split(":")
                self.target_ratios.append(float(ratio.strip()))
                self.ratios.append(0.0)
                self.schemas.append(schema.strip())
                self.queues.append(FIFOQueue())
        self.skipped_action_schemas = set()
        self.num_grounded_actions = 0
    def __bool__(self):
        return any(queue for queue in self.queues)
    __nonzero__ = __bool__
    def get_final_queue(self):
        return list(itertools.chain.from_iterable(q.get_final_queue() for q in self.queues))
    def print_info(self):
        print("Using SchemaRatio priority queue for actions.")
    def print_stats(self):
        for i in range(len(self.schemas)):
            print(f"{self.queues[i].get_num_grounded_actions()} actions grounded for schema {self.schemas[i]}; target ratio: {self.target_ratios[i]}, final ratio: {self.ratios[i]}")
        for schema in self.skipped_action_schemas:
            print(f"WARNING: Action schema {schema} did not appear in given ratios file, so was pruned completely.")
    def get_hard_action_if_exists(self, is_hard_action):
        for i in range(len(self.queues)):
            action = self.queues[i].get_hard_action_if_exists(is_hard_action)
            if (action):
                return [action]
        return []
    def notify_new_hard_actions(self, schemas):
        for schema in schemas:
            if (schema in self.schemas):
                self.queues[self.queues.index(schema)].notify_new_hard_actions(schemas)
    def push(self, action):
        if (not action.predicate.name in self.schemas):
            self.skipped_action_schemas.add(action.predicate.name)
        else:
            index = self.schemas.index(action.predicate.name)
            self.queues[index].push(action)
            if (not options.plan_ratios):
                # ratio is the number of grounded actions of schema X over the total number of actions of X in the queue (grounded or not)
                self.ratios[index] = self.queues[index].get_num_grounded_actions() / self.queues[index].get_num_actions()
    def pop(self):
        self.num_grounded_actions += 1
        next = max([(self.target_ratios[i] - self.ratios[i], i) if self.queues[i] else (-inf, i) for i in range(len(self.ratios))], key=lambda item:item[0])[1]
        if (options.plan_ratios):
            # ratio is the number of grounded actions of schema X over the total number of grounded actions of all schemas
            self.ratios[next] = (self.queues[next].get_num_grounded_actions() + 1.0) / self.num_grounded_actions
            for i in range(len(self.ratios)):
                if (i != next):
                    self.ratios[i] = self.queues[i].get_num_grounded_actions() / self.num_grounded_actions
        else:
            self.ratios[next] = (self.queues[next].get_num_grounded_actions() + 1.0) / self.queues[next].get_num_actions()
        return self.queues[next].pop()
            
class NoveltyFIFOQueue(PriorityQueue):
    def __init__(self):
        self.novel_action_queue = []
        self.novel_queue_pos = 0
        self.novel_last_hard_index = 0
        self.closed_novel_actions = []
        self.non_novel_action_queue = FIFOQueue()
        self.num_novel_actions_grounded = 0
        self.num_non_novel_actions_grounded = 0
        self.novelty = NoveltyEvaluator()
    def __bool__(self):
        if (self.novel_queue_pos < len(self.novel_action_queue)):
            return True
        if (self.non_novel_action_queue):
            return True
        return False
    __nonzero__ = __bool__
    def get_final_queue(self):
        return self.non_novel_action_queue.get_final_queue() + self.closed_novel_actions
    def print_info(self):
        print("Using novelty FIFO priority queue for actions.")
    def print_stats(self):
        print("Grounded %d novel actions" % self.num_novel_actions_grounded)
        print("Grounded %d non-novel actions" % self.num_non_novel_actions_grounded)
    def get_hard_action_if_exists(self, is_hard_action):
        if (self.novel_last_hard_index < self.novel_queue_pos):
            self.novel_last_hard_index = self.novel_queue_pos
        for i in range(self.novel_last_hard_index, len(self.novel_action_queue)):
            action = self.novel_action_queue[i]
            if (is_hard_action(action)):
                self.closed_novel_actions.append(action)
                self.num_novel_actions_grounded += 1
                del self.novel_action_queue[i]
                return [action]
            self.novel_last_hard_index += 1
        hard_actions = self.non_novel_action_queue.get_hard_action_if_exists(is_hard_action)
        if (hard_actions):
            self.num_non_novel_actions_grounded += len(hard_actions)
        return hard_actions
    def notify_new_hard_actions(self, schemas):
        self.non_novel_action_queue.notify_new_hard_actions(schemas)
        self.novel_last_hard_index = self.novel_queue_pos
    def push(self, action):
        if (self.novelty.calculate_novelty(action) == 0):
            self.novel_action_queue.append(action)
        else:
            self.non_novel_action_queue.push(action)
    def pop(self):
        while (self.novel_queue_pos < len(self.novel_action_queue)):
            action = self.novel_action_queue[self.novel_queue_pos]
            self.novel_queue_pos += 1
            if (self.novelty.calculate_novelty(action) == 0):
                self.novelty.update_novelty(action)
                self.num_novel_actions_grounded += 1
                self.closed_novel_actions.append(action)
                return action
            else:
                self.non_novel_action_queue.push(action)
        # removed all actions from novel queue
        assert(self.novel_queue_pos >= len(self.novel_action_queue))
        action = self.non_novel_action_queue.pop()
        self.num_non_novel_actions_grounded += 1
        return action
    
class RoundRobinNoveltyQueue(PriorityQueue):
    def __init__(self):
        self.novelty = NoveltyEvaluator()
        self.schemas = []
        self.current = 0
        self.queues = []
        self.num_grounded_actions = []
        self.closed = []
        self.hard_closed = defaultdict(list)
    def __bool__(self):
        return any(queue for queue in self.queues)
    __nonzero__ = __bool__
    def get_final_queue(self):
        res = self.closed
        for list in self.hard_closed.values():
            res += list
        return res
    def print_info(self):
        print("Using round-robin novelty priority queue for actions.")
    def print_stats(self):
        for i in range(len(self.num_grounded_actions)):
            print("%d actions grounded for schema %s" % (self.num_grounded_actions[i], self.schemas[i]))
    def get_hard_action_if_exists(self, is_hard_action):
        for i in range(len(self.queues)):
            actions = self.queues[i].get_hard_action_if_exists(is_hard_action)
            res = []
            if (len(actions) > 0):
                for action in actions:
                    if (not action in self.hard_closed[action.predicate.name]):
                        res.append(action)
                        self.hard_closed[action.predicate.name].append(action)
                        self.num_grounded_actions[i] += 1
                return res
        return []
    def notify_new_hard_actions(self, schemas):
        for queue in self.queues:
            queue.notify_new_hard_actions()
    def push(self, action):
        novelty = self.novelty.calculate_novelty(action)
        if (not action.predicate.name in self.schemas):
            self.schemas.append(action.predicate.name)
            self.queues.append(SortedHeapQueue())
            self.num_grounded_actions.append(0)
        self.queues[self.schemas.index(action.predicate.name)].push(action, novelty)
    def pop(self):
        while True:
            self.current = (self.current + 1) % len(self.schemas)
            while (self.queues[self.current]):
                novelty_old, action = self.queues[self.current].pop_entry()
                if (action in self.hard_closed[action.predicate.name]):
                    continue
                if (novelty_old == 0):
                    novelty_new = self.novelty.calculate_novelty(action)
                    if (novelty_new != novelty_old):
                        self.queues[self.current].push(action, novelty_new)
                        continue
                self.novelty.update_novelty(action)
                self.closed.append(action)
                self.num_grounded_actions[self.current] += 1
                return action
                    
class RoundRobinEvaluatorQueue(PriorityQueue):
    def __init__(self, model, info):
        # TODO get schemas from model
        self.model = model
        self.schemas = []
        self.current = 0
        self.queues = []
        self.num_grounded_actions = []
        self.closed = []
        self.info = info
        self.batch_eval = options.batch_evaluation
        self.non_evaluated_actions = []
        self.has_actions = False
    def __bool__(self):
        return self.has_actions
    __nonzero__ = __bool__
    def get_final_queue(self):
        return self.closed
    def print_info(self):
        print("Using trained round-robin priority queue for actions.")
        if (self.batch_eval):
            print("Actions are evaluated in batches.")
        print(self.info)
    def print_stats(self):
        self.model.print_stats()
        for i in range(len(self.num_grounded_actions)):
            print("%d actions grounded for schema %s" % (self.num_grounded_actions[i], self.schemas[i]))
    def get_hard_action_if_exists(self, is_hard_action):
        # TODO support for batch evaluation
        for i in range(len(self.queues)):
            action = self.queues[i].get_hard_action_if_exists(is_hard_action)
            if (action):
                self.has_actions = False
                self.closed.append(action)
                self.num_grounded_actions[i] += 1
                if (self.queues[i]):
                    self.has_actions = True
                else:
                    self.__check_has_action()
                return action
        return None
    def notify_new_hard_actions(self):
        for queue in self.queues:
            queue.notify_new_hard_actions()
    def push(self, action):
        # TODO think of a more efficient way to implement this; 
        # can we obtain all schemas from the task initially?
        self.has_actions = True
        if (not action.predicate.name in self.schemas):
            # first time we see this action schema; init datastructures
            self.schemas.append(action.predicate.name)
            self.num_grounded_actions.append(0)
            if (self.batch_eval):
                self.non_evaluated_actions.append([])
            estimate = self.model.get_estimate(action)
            if (estimate != None):
                self.queues.append(SortedHeapQueue(False))
                self.queues[self.schemas.index(action.predicate.name)].push(action, estimate)
            else:
                self.queues.append(FIFOQueue())
                self.queues[self.schemas.index(action.predicate.name)].push(action)
            return
        if (self.batch_eval):
            if (isinstance(self.queues[self.schemas.index(action.predicate.name)], FIFOQueue)):
                self.queues[self.schemas.index(action.predicate.name)].push(action)
            else:
                self.non_evaluated_actions[self.schemas.index(action.predicate.name)].append(action)
        else:
            estimate = self.model.get_estimate(action)
            if (estimate != None):
                self.queues[self.schemas.index(action.predicate.name)].push(action, estimate)
            else:
                self.queues[self.schemas.index(action.predicate.name)].push(action)
    def __evaluate_batch_schema(self, index):
        actions = self.non_evaluated_actions[index]
        if (actions):
            estimates = self.model.get_estimates(actions)
            self.queues[index].push_list(actions, estimates)
            self.non_evaluated_actions[index] = []
    def __check_has_action(self):
        for q in self.queues:
            if (q):
                self.has_actions = True
                return
        if (self.batch_eval):
            for q in self.non_evaluated_actions:
                if (q):
                    self.has_actions = True
                    return
    def pop(self):
        self.has_actions = False
        while True:
            self.current = (self.current + 1) % len(self.schemas)
            if (self.batch_eval):
                self.__evaluate_batch_schema(self.current)
            if (self.queues[self.current]):
                self.num_grounded_actions[self.current] += 1
                action = self.queues[self.current].pop()
                if (self.queues[self.current]):
                    self.has_actions = True
                else:
                    self.__check_has_action()
                # TODO don't add to closed if there is no model for this schema! 
                # these actions are kept in the corresponding FIFOQueue
                self.closed.append(action)
                return action
            
class RatioEvaluatorQueue(PriorityQueue):
    # TODO implement support for batch evaluation
    def __init__(self, model, info):
        # TODO check with schemas from model
        if (not options.action_schema_ratios):
            sys.exit("Error: need action schema ratios to use this queue. Please specify using --action-schema-ratios")
        with open(options.action_schema_ratios, "r") as ratios:
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
                self.queues.append(FIFOQueue())
                self.num_grounded_actions.append(0)
                self.num_actions.append(0)
        self.model = model
        self.closed = []
        self.skipped_action_schemas = set()
        self.queue_got_push = [False for s in self.schemas]
        self.info = info
        if (options.batch_evaluation):
            print("WARNING: batch evaluation of actions not yet implemented for RatioEvaluatorQueue")
    def __bool__(self):
        return any(q for q in self.queues)
    __nonzero__ = __bool__
    def get_final_queue(self):
        return self.closed
    def print_info(self):
        print("Using trained action-schema ratio priority queue for actions.")
        print(self.info)
    def print_stats(self):
        self.model.print_stats()
        for i in range(len(self.num_grounded_actions)):
            print(f"{self.num_grounded_actions[i]} actions grounded for schema {self.schemas[i]}; target ratio: {self.target_ratios[i]}, final ratio: {self.ratios[i]}")
        for schema in self.skipped_action_schemas:
            print(f"WARNING: Action schema {schema} did not appear in given ratios file, so was pruned completely.")
    def get_hard_action_if_exists(self, is_hard_action):
        for i in range(len(self.queues)):
            action = self.queues[i].get_hard_action_if_exists(is_hard_action)
            if (action):
                self.closed.append(action)
                self.num_grounded_actions[i] += 1
                return action
        return None
    def notify_new_hard_actions(self):
        for queue in self.queues:
            queue.notify_new_hard_actions()
    def push(self, action):
        if (not action.predicate.name in self.schemas):
            self.skipped_action_schemas.add(action.predicate.name)
            return
        estimate = self.model.get_estimate(action)
        index = self.schemas.index(action.predicate.name)
        if (not self.queue_got_push[index]):
            self.queue_got_push[index] = True
            if (estimate != None):
                self.queues[index] = SortedHeapQueue(False)
        if (estimate != None):
            self.queues[index].push(action, estimate)
        else:
            self.queues[index].push(action)
        self.num_actions[index] += 1
        if (not options.plan_ratios):
            self.ratios[index] = self.num_grounded_actions[index] / self.num_actions[index]
    def pop(self):
        next = max([(self.target_ratios[i] - self.ratios[i], i) if self.queues[i] else (-inf, i) for i in range(len(self.ratios))], key=lambda item:item[0])[1]
        self.num_grounded_actions[next] += 1
        action = self.queues[next].pop()
        # TODO don't add to closed if there is no model for this schema! 
        # these actions are kept in the corresponding FIFOQueue
        self.closed.append(action)
        if (options.plan_ratios):
            total_num_grounded = sum(self.num_grounded_actions)
            for i in range(len(self.ratios)):
                self.ratios[i] = self.num_grounded_actions[i] / total_num_grounded
        else:
            self.ratios[next] = self.num_grounded_actions[next] / self.num_actions[next]
        return action

class HardRulesQueue(PriorityQueue):
    def __init__(self, inner_queue, hard_rules_evaluator):
        self.queue = inner_queue
        self.evaluator = hard_rules_evaluator
        self.num_hard_rule_actions = 0
        self.tmp_hard_actions = []
    def __bool__(self):
        return bool(self.queue)
    __nonzero__ = __bool__
    def get_final_queue(self):
        return self.queue.get_final_queue()
    def print_info(self):
        print("Using hard-rule priority queue for actions with the following inner queue and hard-rule evaluator:")
        self.queue.print_info()
        self.evaluator.print_info()
    def print_stats(self):
        print("Number actions from hard rules:", self.num_hard_rule_actions)
        self.queue.print_stats()
    def push(self, action):
        self.queue.push(action)
    def pop(self):
        if (len(self.tmp_hard_actions) == 0):
            self.tmp_hard_actions = self.queue.get_hard_action_if_exists(self.evaluator.is_hard_action)
        if (len(self.tmp_hard_actions) > 0):
            ground_action = self.tmp_hard_actions.pop()
            self.num_hard_rule_actions += 1
        else:
            ground_action = self.queue.pop()
        target_schemas = self.evaluator.notify_action(ground_action)
        if (len(target_schemas) > 0):
            self.queue.notify_new_hard_actions(target_schemas)
        return ground_action
        
        
