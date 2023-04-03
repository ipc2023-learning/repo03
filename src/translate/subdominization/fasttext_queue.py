from .queue_factory import PriorityQueue
from .priority_queue import FIFOQueue, SortedHeapQueue
import options

from collections import defaultdict
import fasttext
import itertools
from math import inf
import mlflow.sklearn
import numpy as np
import os
import pandas as pd
import random
import re
from sklearn.base import BaseEstimator, TransformerMixin


def get_lines_from_file(file):
    with open(file) as f:
        return [line.replace("\n", "") for line in f]


def get_useful_facts_from_file(useful_facts_file):
    facts = get_lines_from_file(useful_facts_file)
    facts = [re.sub(r"[,]", ", ", s) for s in facts]
    facts = [re.sub(r"[(]", " ", s) for s in facts]
    return [re.sub(r"[),]", "", s) for s in facts]

def get_fasttext_evaluator(model_path, useful_facts_file):
    useful_facts = get_useful_facts_from_file(useful_facts_file)
    if options.optimized_fasttext:
        return FastTextEvaluatorOptimized(model_path, useful_facts)
    else:
        return FastTextEvaluator(model_path, useful_facts)


class FastTextQueue(PriorityQueue):
    def __init__(self, useful_facts_file, model_path, classifiers_path):
        if (not os.path.isdir(classifiers_path)):
            exit(f"Error: given --trained-model-folder is not a folder: {classifiers_path}")

        self.queue = SortedHeapQueue(False)
        self.closed = []

        self.evaluator = get_fasttext_evaluator(model_path, useful_facts_file)

        # TODO change to list of classifiers/batches to avoid dict lookup
        self.clf = {}
        self.schema_batches = {}
        for schema_folder in os.listdir(classifiers_path):
            if (os.path.isdir(os.path.join(classifiers_path, schema_folder))):
                self.clf[schema_folder] = mlflow.sklearn.load_model(
                    os.path.join(classifiers_path, schema_folder))
                self.schema_batches[schema_folder] = []

    def __bool__(self):
        return bool(self.queue) or any(b for b in self.schema_batches.values())

    __nonzero__ = __bool__

    def get_final_queue(self):
        return self.closed

    def print_info(self):
        print("Using FastTextQueue priority queue for actions.")

    def push(self, action):
        # TODO have a single-evaluation version?
        schema = action.predicate.name
        if (not schema in self.schema_batches):
            # TODO collect these action schemas? skip them?
            self.queue.push(action, 1.0)
        else:
            self.schema_batches[schema].append(action)

    def pop(self):
        for schema, batch in self.schema_batches.items():
            if (not batch):
                continue

            self.queue.push_list(batch, self.evaluator.get_estimates(batch, self.clf[schema]))
            self.schema_batches[schema] = []

        a = self.queue.pop()
        self.closed.append(a)
        return a


class RoundRobinFastTextQueue(PriorityQueue):
    def __init__(self, useful_facts_file, model_path, classifiers_path):
        if (not os.path.isdir(classifiers_path)):
            exit(f"Error: given --trained-model-folder is not a folder: {classifiers_path}")

        self.closed = []

        self.evaluator = get_fasttext_evaluator(model_path, useful_facts_file)

        # TODO change to list of classifiers/batches/.. to avoid dict lookup
        self.clf = {}
        self.schema_batches = {}
        self.queues = {}
        self.schemas = []
        self.trained_schemas = []
        for schema_folder in os.listdir(classifiers_path):
            if (os.path.isdir(os.path.join(classifiers_path, schema_folder))):
                # none is our magic word here, hope no action is called like that
                assert (schema_folder != "none")
                self.clf[schema_folder] = mlflow.sklearn.load_model(
                    os.path.join(classifiers_path, schema_folder))
                self.schema_batches[schema_folder] = []
                self.queues[schema_folder] = SortedHeapQueue(False)
                self.schemas.append(schema_folder)
                self.trained_schemas.append(schema_folder)

        self.current = 0

    def __bool__(self):
        return any(q for q in self.queues.values()) or any(b for b in self.schema_batches.values())

    __nonzero__ = __bool__

    def get_final_queue(self):
        return self.closed + list(itertools.chain.from_iterable(fifo_q.get_final_queue() for schema, fifo_q in self.queues.items() if schema not in self.trained_schemas))

    def print_info(self):
        print("Using RoundRobinFastTextQueue priority queue for actions.")

    def print_stats(self):
        for schema in sorted(s for s in self.schemas if s not in self.trained_schemas):
            print(f"WARNING: Action schema {schema} did not have a trained model.")

    def push(self, action):
        # TODO have a single-evaluation version?
        schema = action.predicate.name
        if (not schema in self.trained_schemas):
            if (not schema in self.schemas):
                self.schemas.append(schema)
                self.queues[schema] = FIFOQueue()
                self.schema_batches[schema] = None
            self.queues[schema].push(action)
        else:
            self.schema_batches[schema].append(action)

    def pop(self):
        while True:
            current_schema = self.schemas[self.current]
            self.current = (self.current + 1) % len(self.schemas)

            batch = self.schema_batches[current_schema]

            if (batch):
                self.queues[current_schema].push_list(batch, self.evaluator.get_estimates(batch, self.clf[current_schema]))
                self.schema_batches[current_schema] = []

            if (not self.queues[current_schema]):
                continue

            a = self.queues[current_schema].pop()
            if (current_schema in self.trained_schemas):
                self.closed.append(a)
            return a


class RatioFastTextQueue(PriorityQueue):
    def __init__(self, useful_facts_file, model_path, classifiers_path):
        if (not options.action_schema_ratios):
            exit("Error: need action schema ratios to use this queue. Please specify using --action-schema-ratios")

        with open(options.action_schema_ratios, "r") as ratios:
            self.target_ratios = []
            self.ratios = []
            self.schemas = []
            self.num_grounded_actions = []
            self.num_actions = []
            for line in ratios:
                schema, ratio = line.split(":")
                self.schemas.append(schema.strip())
                self.target_ratios.append(float(ratio.strip()))
                self.ratios.append(0.0)
                self.num_grounded_actions.append(0)
                self.num_actions.append(0)

        self.evaluator = get_fasttext_evaluator(model_path, useful_facts_file)

        self.skipped_action_schemas = set()

        # TODO change to list of classifiers/batches/.. to avoid dict lookup
        self.queues = defaultdict(FIFOQueue)
        self.clf = {}
        self.schema_batches = {}
        self.trained_schemas = set()
        for schema in os.listdir(classifiers_path):
            if (os.path.isdir(os.path.join(classifiers_path, schema))):
                if (schema not in self.schemas):
                    self.skipped_action_schemas.add(schema)
                    continue
                self.trained_schemas.add(schema)
                self.clf[schema] = mlflow.sklearn.load_model(
                    os.path.join(classifiers_path, schema))
                self.schema_batches[schema] = []
                self.queues[schema] = SortedHeapQueue(False)

        self.closed = []
        self.total_num_grounded = 0

    def __bool__(self):
        return any(q for q in self.queues.values()) or any(b for b in self.schema_batches.values())
    __nonzero__ = __bool__

    def get_final_queue(self):
        return self.closed + list(itertools.chain.from_iterable(fifo_q.get_final_queue() for schema, fifo_q in self.queues.items() if schema not in self.trained_schemas))

    def print_info(self):
        print("Using fasttext action-schema ratio priority queue for actions.")

    def print_stats(self):
        for i in range(len(self.num_grounded_actions)):
            print(f"{self.num_grounded_actions[i]} actions grounded for schema {self.schemas[i]}; target ratio: {self.target_ratios[i]}, final ratio: {self.ratios[i]}")
        for schema in self.skipped_action_schemas:
            print(f"WARNING: Action schema {schema} did not appear in given ratios file, so was pruned completely.")
    def push(self, action):
        schema = action.predicate.name
        if (not schema in self.schemas):
            self.skipped_action_schemas.add(schema)
            return

        if (schema in self.trained_schemas):
            self.schema_batches[schema].append(action)
        else:
            self.queues[schema].push(action)

        index = self.schemas.index(schema)
        self.num_actions[index] += 1
        if (not options.plan_ratios):
            self.ratios[index] = self.num_grounded_actions[index] / self.num_actions[index]

    def _has_action(self, schema):
        return (self.schema_batches[schema] or self.queues[schema]) if schema in self.trained_schemas else bool(self.queues[schema])

    def pop(self):
        next = max([(self.target_ratios[i] - self.ratios[i], i)
                    if self._has_action(self.schemas[i]) else (-inf, i)
                    for i in range(len(self.ratios))], key=lambda item: item[0])[1]
        next_schema = self.schemas[next]
        has_model = next_schema in self.trained_schemas

        if (has_model):
            batch = self.schema_batches[next_schema]

            if (batch):
                self.queues[next_schema].push_list(batch, self.evaluator.get_estimates(batch, self.clf[next_schema]))
                self.schema_batches[next_schema] = []

        action = self.queues[next_schema].pop()
        if (has_model):
            self.closed.append(action)

        self.num_grounded_actions[next] += 1
        self.total_num_grounded += 1
        if (options.plan_ratios):
            for i in range(len(self.ratios)):
                self.ratios[i] = self.num_grounded_actions[i] / self.total_num_grounded
        else:
            self.ratios[next] = self.num_grounded_actions[next] / self.num_actions[next]

        return action


class FastTextEvaluator:
    def __init__(self, model_path, useful_facts) -> None:
        self.windowizer = Windowizer(
            w_size=4, step=3, remain_order=False, nof_samples_p_win=25)
        
        good_ops_model = os.path.join(model_path, "model_cbow_0.05_100_30_4_4_good_operators_sentence.bin")
        relaxed_plan_model = os.path.join(model_path, "model_cbow_0.05_100_30_4_4_relaxed_plan_sentence.bin")

        self.ffvectorizer = FastTextVectorizerC(good_ops_model, relaxed_plan_model)
        
        self.useful_facts = useful_facts

    def get_estimates(self, actions, classifier):
        batch_to_predict = pd.DataFrame({
            "problem_id": [0] * len(actions),
            "facts": [self.useful_facts] * len(actions),
            "query_action": [f"({a.predicate.name} {' '.join(a.args)})" for a in actions]
        })

        windows = self.windowizer.fit_transform(batch_to_predict)
        emb_windows = self.ffvectorizer.fit_transform(windows)

        X_test = emb_windows.drop(columns=["problem_id", "query_id"])
        X_test_arr = np.hstack([np.vstack(X_test[feature])
                                for feature in X_test.columns])

        y_prob = classifier.predict_proba(X_test_arr)[:, 1]

        # These probabilities correspond to a pair (windowed facts, query_action). For
        # each query action, we agreggate all its corresponding window probabilities
        # using the `max` function.

        emb_windows["y_prob"] = y_prob

        scores = emb_windows[["problem_id", "query_id", "y_prob"]].groupby(
            ["problem_id", "query_id"]).max()
        
        return scores.values.flatten().tolist()
    

class FastTextEvaluatorOptimized:
    def __init__(self, model_path, useful_facts) -> None:
        good_ops_model = os.path.join(model_path, "model_cbow_0.05_100_30_4_4_good_operators_sentence.bin")
        relaxed_plan_model = os.path.join(model_path, "model_cbow_0.05_100_30_4_4_relaxed_plan_sentence.bin")

        self.windowizer = FastWindowizer(useful_facts, good_ops_model, relaxed_plan_model)

    def get_estimates(self, actions, classifier):
        X_emb = self.windowizer.fit_transform([f"({a.predicate.name} {' '.join(a.args)})" for a in actions])

        y_prob = classifier.predict_proba(X_emb)[:, 1]

        num_windows = int(len(y_prob) / len(actions))

        probs = [max(y_prob[i:i+num_windows]) for i in range(0, len(y_prob), num_windows)]

        return probs


class FastWindowizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        useful_facts,
        ft_action_path,
        ft_facts_path,
        w_size=4,
        step=3,
        nof_samples_p_win=25,
    ):
        # Load fasttext model
        self.ft_model_action = fasttext.load_model(ft_action_path)
        self.ft_model_facts = fasttext.load_model(ft_facts_path)
        # Define data structures
        self.segments = []
        self.str_facts = useful_facts
        self.emb_facts = []
        # Windowizing parameters
        self.w_size = w_size
        self.step = step
        self.nof_samples_p_win = nof_samples_p_win
        # Preprocess facts file
        self.preprocess()

    def preprocess(self):
        """Precomputes embeddings"""
        # Get proposition types for segment identification
        schs = [rf.split()[0] for rf in self.str_facts]
        # Define
        self.emb_facts = np.zeros(
            (len(self.str_facts), self.ft_model_facts.get_dimension())
        )
        self.num_words = np.zeros(len(self.str_facts))

        # Find segments and compute embeddings
        j = 0
        for i, fact in enumerate(self.str_facts):
            # If new segment has started, log it
            if schs[j] != schs[i]:
                self.segments.append((j, i - 1))
                j = i
            # Split facts into words
            words = ["("]
            words += fact.split()
            words.append(")")
            # Compute embeddings and count words
            for w in words:
                vec = self.ft_model_facts.get_word_vector(w)
                self.emb_facts[i, :] += vec / np.linalg.norm(vec)
                self.num_words[i] += 1
        # Log last segment
        self.segments.append((j, len(self.str_facts) - 1))
        # Prevent undefined behavior when w_size < len(segments)
        self.w_size = min(self.w_size, len(self.segments))

    def fit(self, X):
        return self

    def transform(self, X):
        """Fast window generation routine"""
        # Compute window embeddings
        w_embs, N = self.compute_window_embedding(len(X))
        # Compute action embeddings
        a_embs = self.compute_action_embedding(X, N)
        # Combine
        return np.hstack([w_embs, a_embs])

    def compute_window_embedding(self, num_actions):
        """Compute embeddings for windows"""
        # Compute window segment indices
        max_idx = len(self.segments) - self.w_size
        base = np.arange(self.w_size)
        shift = np.arange(0, max_idx + 1, self.step)
        shift = np.append(shift, max_idx) if max_idx not in shift else shift
        sub_windows = np.expand_dims(base, 0) + np.expand_dims(shift, 0).T
        sub_windows_rep = np.repeat(
            sub_windows, self.nof_samples_p_win, axis=0)
        N = len(sub_windows_rep)
        sub_windows_rep = np.repeat(sub_windows_rep, num_actions, axis=0)
        # Map segment indices to indices within the fact list
        segments_array = np.array(self.segments)
        s_left = segments_array[:, 0]
        s_right = segments_array[:, 1]
        s_left = s_left[sub_windows_rep]
        s_right = s_right[sub_windows_rep]
        M = len(self.segments) if len(self.segments) < self.w_size else self.w_size
        w_np = s_left + np.random.rand(len(s_right), M) * (
            s_right - s_left
        )
        w_np = np.round(w_np).astype(int)
        # Aggregate embeddings and normalize
        w_embs = np.sum(self.emb_facts[w_np], axis=1)
        w_num_words = np.sum(self.num_words[w_np], axis=1)
        w_embs = w_embs / w_num_words.reshape((-1, 1))
        return w_embs, N

    def compute_action_embedding(self, actions, reps):
        """Compute repeated embeddings for actions"""
        a_embs = np.zeros(
            (len(actions), reps, self.ft_model_action.get_dimension()))
        for i, action in enumerate(actions):
            vec = self.ft_model_action.get_sentence_vector(action)
            a_embs[i, :, :] = np.repeat(vec.reshape((1, -1)), reps, axis=0)
        a_embs = a_embs.reshape(-1, self.ft_model_action.get_dimension())
        return a_embs


class BaseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, data):
        return self

    def transform(self, data):
        return data


class FastTextVectorizerC(BaseTransformer):
    def __init__(self, ft_action_path, ft_facts_path):
        assert(os.path.exists(ft_facts_path))
        assert(os.path.exists(ft_action_path))

        self.models = [fasttext.load_model(ft_facts_path),
                       fasttext.load_model(ft_action_path)]

    def get_action_vector(self, action):
        if len(self.models) == 2:
            return self.models[1].get_sentence_vector(action)
        else:
            return self.models[0].get_sentence_vector(action)

    def get_fact_vector(self, fact):
        return self.models[0].get_sentence_vector(fact)

    def get_sequence_vector(self, plan):
        w_sep_plan = "( " + " ) ( ".join(plan) + " )"
        return self.models[0].get_sentence_vector(w_sep_plan)

    def fit(self, df):
        return self

    def transform(self, df):
        df = df.copy()
        df["facts"] = df["facts"].apply(self.get_sequence_vector)
        df["query_action"] = df["query_action"].apply(self.get_action_vector)
        return df


class Windowizer(BaseTransformer):
    def __init__(self, w_size=3, step=3, remain_order=True, nof_samples_p_win=None):
        self.w_size = w_size
        self.step = step
        self.remain_order = remain_order
        self.nof_samples_p_win = nof_samples_p_win

    def extract_windows_segmented(self, array):
        segments = []
        l = 0
        schs = [rf.split()[0] for rf in array]
        for r in range(1, len(schs)):
            if schs[l] != schs[r]:
                segments.append((l, r))
                l = r
        segments.append((l, len(schs)))

        arr_np = np.array(array, dtype="object")

        if len(segments) < self.w_size:
            idxs = [int(l + (r - l) * random.random()) for l, r in segments]
            return arr_np[idxs]

        max_idx = len(segments) - self.w_size
        base = np.arange(self.w_size)
        shift = np.arange(0, max_idx + 1, self.step)
        shift = np.append(shift, max_idx) if max_idx not in shift else shift
        sub_windows = np.expand_dims(base, 0) + np.expand_dims(shift, 0).T
        sub_windows_rep = np.repeat(sub_windows, self.nof_samples_p_win, axis=0)
        segments_array = np.array(segments)

        windows = []
        for w in sub_windows_rep:
            windows.append(
                [int(l + (r - l) * random.random())
                 for l, r in segments_array[w]]
            )

        windows_np = np.array(windows)
        return arr_np[windows_np].tolist()

    def extract_windows_vectorized(self, array):
        """
        Vectorized function to extract attribute windows
        ----------
        Inputs:
        - array (list): Attribute array from which windows are to be extracted
        - window_size (int): Size of attribute window
        - step (int): Step size between adjacent windows

        Outputs:
        - List of lists for each attribute window
          ----------
        NOTE: If the length of the attribute is smaller than the window_size the
        whole attribute is returned.
        """

        # Handle small length edgecase
        if len(array) < self.w_size:
            return [array]
        # Convert array to np object array to take advantage of indexing
        arr_np = np.array(array, dtype="object")
        # Compute starting point of last window required
        max_idx = len(arr_np) - self.w_size
        # Construct base and shift for vectorized computation
        base = np.arange(self.w_size)
        shift = np.arange(0, max_idx + 1, self.step)
        # Include index for last window if not exactly divisible
        shift = np.append(shift, max_idx) if max_idx not in shift else shift
        # Construct window indexes
        sub_windows = np.expand_dims(base, 0) + np.expand_dims(shift, 0).T
        # Extract and return
        return arr_np[sub_windows].tolist()

    def transform(self, query_df):

        problems = query_df[["problem_id", "facts"]].drop_duplicates(
            subset="problem_id"
        )

        query_df = query_df.copy()
        if "query_id" not in query_df:
            query_df["query_id"] = query_df.groupby("problem_id").cumcount()

        # Build windows of all the relaxed plans in @df
        w_attrs = []
        ids = []
        extract_windows = (
            self.extract_windows_segmented
            if self.nof_samples_p_win is not None
            else self.extract_windows_vectorized
        )
        for id, relaxed_plan in zip(problems.problem_id, query_df.facts):
            w_attr = extract_windows(relaxed_plan)
            w_attrs.extend(w_attr)
            ids.extend([id] * len(w_attr))

        # Convert windowed elements to strings
        w_attrs_strs = [[str(act) for act in w_attr] for w_attr in w_attrs]

        # Pad attributes to same length using empty strings
        for w_attr_strs in w_attrs_strs:
            L = len(w_attr_strs)
            if L != self.w_size:
                w_attr_strs.extend([""] * (self.w_size - L))

        # Definition of data dictionary for problem df
        w_problem_data = {
            "problem_id": ids,
            "facts": w_attrs_strs,
        }

        # Construct problem df
        w_problem_df = pd.DataFrame(w_problem_data)

        if self.remain_order:
            w_order = np.zeros(len(w_problem_df))
            for id in w_problem_df["problem_id"].unique():
                w_in_problem = w_problem_df["problem_id"] == id
                nof_ws = np.sum(w_in_problem)
                w_order[w_in_problem] = np.arange(nof_ws)

            w_problem_df["window_order"] = w_order

        data_df = pd.merge(
            w_problem_df, query_df.drop(columns="facts"), on="problem_id"
        )
        return data_df
