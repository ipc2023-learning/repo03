import options
import timers

import sys


class PriorityQueue:
    def __init__(self):
        raise

    def get_final_queue(self):
        pass

    def has_good_actions(self):
        return False

    def get_num_grounded_actions(self):
        pass

    def get_num_actions(self):
        pass

    def print_stats(self):
        print("no statistics available")


def get_action_queue_from_options(task=None):
    # TODO check existing/valid options in constructors
    import subdominization.priority_queue as pq
    name = options.grounding_action_queue_ordering.lower()
    if name == "fifo":
        return pq.FIFOQueue()
    elif name == "lifo":
        return pq.LIFOQueue()
    elif name == "random":
        return pq.EvaluatorQueue(pq.RandomEvaluator(), "Using random action priority.")
    elif name in ["ipc23-single-queue", "ipc23-round-robin", "ipc23-ratio"]:
        from .ipc23_queue import IPC23SingleQueue, IPC23RoundRobinQueue, IPC23RatioQueue
        if name == "ipc23-single-queue":
            return IPC23SingleQueue(task, options)
        elif name == "ipc23-round-robin":
            return IPC23RoundRobinQueue(task, options)
        elif name == "ipc23-ratio":
            return IPC23RatioQueue(task, options)
    elif name in ["trained", "roundrobintrained"]:
        from .model import TrainedModel
        if not options.trained_model_folder:
            sys.exit("Error: need trained model to use this queue. Please specify using --trained-model-folder")
        if not task:
            sys.exit("Error: no task given")
        timer = timers.Timer()
        model = TrainedModel(options.trained_model_folder, task)
        if name == "trained":
            return pq.EvaluatorQueue(model, f"Loaded trained model from {options.trained_model_folder} {str(timer)}")
        elif name == "roundrobintrained":
            return pq.RoundRobinEvaluatorQueue(model, f"Loaded trained model from {options.trained_model_folder} {str(timer)}")
    elif name in ["aleph", "roundrobinaleph"]:
        from .rule_evaluator_aleph import RuleEvaluatorAleph
        if not options.aleph_model_file:
            sys.exit("Error: need trained model to use this queue. Please specify using --aleph-model-file")
        if not task:
            sys.exit("Error: no task given")
        timer = timers.Timer()
        with open(options.aleph_model_file, "r") as aleph_rules:
            model = RuleEvaluatorAleph(aleph_rules.readlines(), task)
        if name == "aleph":
            return pq.EvaluatorQueue(model, f"Loaded aleph model from {options.aleph_model_file} {str(timer)}")
        elif name == "roundrobinaleph":
            return pq.RoundRobinEvaluatorQueue(model, f"Loaded aleph model from {options.aleph_model_file} {str(timer)}")
    elif name == "roundrobin":
        return pq.RoundRobinQueue()
    elif name == "noveltyfifo":
        return pq.NoveltyFIFOQueue()
    elif name == "roundrobinnovelty":
        return pq.RoundRobinNoveltyQueue()
    elif name == "actionsfromfile":
        return pq.ActionsFromFileQueue()
    elif name == "ratio":
        return pq.RatioQueue()
    elif name == "ratiotrained":
        from .model import TrainedModel
        if not options.trained_model_folder:
            sys.exit("Error: need trained model to use this queue. Please specify using --trained-model-folder")
        if not task:
            sys.exit("Error: no task given")
        timer = timers.Timer()
        model = TrainedModel(options.trained_model_folder, task)
        return pq.RatioEvaluatorQueue(model, f"Loaded trained model from {options.trained_model_folder} {str(timer)}")
    elif name == "ratioaleph":
        from .rule_evaluator_aleph import RuleEvaluatorAleph
        if not options.aleph_model_file:
            sys.exit("Error: need trained model to use this queue. Please specify using --aleph-model-file")
        if not task:
            sys.exit("Error: no task given")
        timer = timers.Timer()
        with open(options.aleph_model_file, "r") as aleph_rules:
            model = RuleEvaluatorAleph(aleph_rules.readlines(), task)
        return pq.RatioEvaluatorQueue(model, f"Loaded trained model from {options.aleph_model_file} {str(timer)}")
    elif name == "policy":
        from .policy_queue import PolicyQueue
        if not options.policy_file:
            sys.exit("Error: need policy to use this queue. Please specify using --policy-file")
        if not task:
            sys.exit("Error: no task given")
        return PolicyQueue(task, options.policy_file)
    elif name in ["fasttext", "roundrobinfasttext", "ratiofasttext"]:
        if not options.trained_model_folder:
            # TODO this is shared among several queues => make this a helper function
            sys.exit("Error: need trained classifier models to use this queue. Please specify using --trained-model-folder")
        if not options.fasttext_model:
            sys.exit("Error: need trained fasttext model to use this queue. Please specify using --fasttext-model")
        if not options.relaxed_plan_file:
            sys.exit("Error: need relaxed plan file to use this queue. Please specify using --relaxed-plan-file")
        if name == "fasttext":
            from .fasttext_queue import FastTextQueue
            return FastTextQueue(options.relaxed_plan_file, options.fasttext_model, options.trained_model_folder)
        elif name == "roundrobinfasttext":
            from .fasttext_queue import RoundRobinFastTextQueue
            return RoundRobinFastTextQueue(options.relaxed_plan_file, options.fasttext_model, options.trained_model_folder)
        elif name == "ratiofasttext":
            from .fasttext_queue import RatioFastTextQueue
            return RatioFastTextQueue(options.relaxed_plan_file, options.fasttext_model, options.trained_model_folder)
    else:
        sys.exit(f"Error: unknown queue type: {name}")
    assert False
