from .queue_factory import PriorityQueue

import pddl.conditions
import dlplan


class Action:
    def __init__(self, action):
        self.action = action
        self.is_grounded = False

        self.params_map = {}

        self.precondition_atom_ids = set()
        self.add_atom_ids = set()
        self.del_atom_ids = set()

    def _compute_params_map_if_necessary(self):
        if not self.params_map:
            for i, param in enumerate(self.action.predicate.parameters):
                self.params_map[param.name] = self.action.args[i]

    def _compute_add_del_atom_ids_if_necessary(self, instance_info):
        if not self.add_atom_ids and not self.del_atom_ids:
            self._compute_params_map_if_necessary()
            for eff in self.action.predicate.effects:
                # check for non-supported features
                assert eff.condition == pddl.conditions.Truth()
                assert not eff.parameters
                eff_atom_id = instance_info.add_atom(eff.literal.predicate,
                                                     [self.params_map[a] for a in eff.literal.args]).get_index()
                if eff.literal.negated:
                    self.del_atom_ids.add(eff_atom_id)
                else:
                    self.add_atom_ids.add(eff_atom_id)

    def get_precondition_atom_ids(self, instance_info):
        if not self.precondition_atom_ids:
            self._compute_params_map_if_necessary()
            if isinstance(self.action.predicate.precondition, pddl.conditions.Atom):
                cond = self.action.predicate.precondition
                assert not cond.negated  # not supported
                cond_atom = instance_info.add_atom(cond.predicate, [self.params_map[a] for a in cond.args])
                self.precondition_atom_ids.add(cond_atom.get_index())
            else:
                assert isinstance(self.action.predicate.precondition, pddl.conditions.Conjunction)
                for cond in self.action.predicate.precondition.parts:
                    assert not cond.negated  # not supported
                    cond_atom = instance_info.add_atom(cond.predicate, [self.params_map[a] for a in cond.args])
                    self.precondition_atom_ids.add(cond_atom.get_index())
        return self.precondition_atom_ids

    def get_add_atom_ids(self, instance_info):
        self._compute_add_del_atom_ids_if_necessary(instance_info)
        return self.add_atom_ids

    def get_del_atom_ids(self, instance_info):
        self._compute_add_del_atom_ids_if_necessary(instance_info)
        return self.del_atom_ids

    def get_name(self):
        self._compute_params_map_if_necessary()
        return f"{self.action.predicate.name}({', '.join(self.params_map[p.name] for p in self.action.predicate.parameters)})"

    def dump(self):
        self._compute_params_map_if_necessary()
        print(f"{self.action.predicate.name}({', '.join(self.params_map[p.name] for p in self.action.predicate.parameters)}): "
              f"parameters [{', '.join(x.name for x in self.action.predicate.parameters)}]; "
              f"instantiations [{', '.join(self.action.args)}]")
        print("preconditions:")
        for cond in self.action.predicate.precondition.parts:
            print(f"\t{'not ' if cond.negated else ''} {cond.predicate}({', '.join(self.params_map[a] for a in cond.args)})")
        print("effects:")
        for eff in self.action.predicate.effects:
            print(f"\t{'not ' if eff.literal.negated else ''}{eff.literal.predicate}({', '.join(self.params_map[a] for a in eff.literal.args)})")


class SGNode:
    def __init__(self, actions, depth, instance_info):
        print(depth)
        pre_to_actions = {}
        self.actions = []
        for action in actions:
            for pre_id in sorted(action.get_precondition_atom_ids(instance_info))[depth:]:
                if pre_id not in pre_to_actions:
                    pre_to_actions[pre_id] = []
                pre_to_actions[pre_id].append(action)
            if not action.get_precondition_atom_ids(instance_info):
                # TODO store ids, not actions
                self.actions.append(action)

        if pre_to_actions:
            preconditions = [(key, pre_to_actions[key]) for key in pre_to_actions]
            preconditions.sort(key=lambda x: len(x[1]), reverse=True)

            self.atom_id = preconditions[0][0]

            self.atom_true_child = SGNode(preconditions[0][1], depth + 1, instance_info)
            self.atom_false_child = SGNode([a for l in preconditions[1:] for a in l[1]], depth, instance_info)

    def add_action(self, action, action_id, num_handled_pre, instance_info):
        if num_handled_pre == len(action.get_precondition_atom_ids(instance_info)):
            self.actions.append(action)  # TODO store ids, not actions
            return
        if self.atom_id in action.get_precondition_atom_ids(instance_info):
            self.atom_true_child.add_action(action, action_id, num_handled_pre + 1)
        else:
            self.atom_false_child.add_action(action, action_id, num_handled_pre)

    def get_applicable_actions(self, state, appl_actions):
        appl_actions += self.actions
        if self.atom_id in state.atom_ids:
            self.atom_true_child.get_applicable_actions(state, appl_actions)
        else:
            self.atom_false_child.get_applicable_actions(state, appl_actions)

class SuccessorGenerator:
    def __init__(self, actions, instance_info):
        self.root_node = SGNode(actions, 0, instance_info)

    def add_action(self, action, action_id, instance_info):
        self.root_node.add_action(action, action_id, 0, instance_info)

    def get_applicable_actions(self, state):
        appl_actions = []
        self.root_node.get_applicable_actions(state, appl_actions)
        return appl_actions


class State:
    def __init__(self, atoms_ids, instance_info):
        self.state = dlplan.State(instance_info, list(atoms_ids))
        self.atom_ids = atoms_ids

    def is_applicable(self, action):
        return action.get_precondition_atom_ids(self.state.get_instance_info()).issubset(self.atom_ids)


class PolicyQueue(PriorityQueue):
    def __init__(self, task, policy_file):
        self.sg = None
        assert(not task.axioms) # not supported
        assert(not task.requirements or
               task.requirements != ":derived-predicates" or
               ":derived-predicates" not in task.requirements)
        assert(isinstance(task.goal, pddl.conditions.Conjunction) or
               isinstance(task.goal, pddl.conditions.Atom))

        ignore_special_predicates = ["="]
        self.vocabulary_info = dlplan.VocabularyInfo()

        types = set()
        for t in task.types:
            #if t.name != "object": TODO check this
            self.vocabulary_info.add_predicate(t.name, 1)
            types.add(t.name)

        for pred in task.predicates:
            if pred.name not in ignore_special_predicates:
                self.vocabulary_info.add_predicate(pred.name, len(pred.arguments))

        for const in task.constants:
            self.vocabulary_info.add_constant(const.name)

        # add goal predicates
        if isinstance(task.goal, pddl.conditions.Atom):
            self.vocabulary_info.add_predicate(f"{task.goal.predicate}_g", len(task.goal.args))
        else:
            for goal in task.goal.parts:
                assert goal.predicate not in ignore_special_predicates
                self.vocabulary_info.add_predicate(f"{goal.predicate}_g", len(goal.args))

        factory = dlplan.SyntacticElementFactory(self.vocabulary_info)

        with open(policy_file, "r") as f:
            self.policy = dlplan.PolicyReader().read("\n".join(f.readlines()), factory)

        self.eval_cache = dlplan.EvaluationCache(len(self.policy.get_boolean_features()),
                                                 len(self.policy.get_numerical_features()))

        self.instance_info = dlplan.InstanceInfo(self.vocabulary_info)

        init_atom_ids = []
        #print("\nInitial state:")
        for init in task.init:  # TODO check if there is some way to obtain static atoms that are not type
            if init.predicate not in ignore_special_predicates:
                if init.predicate in types:
                    init_atom = self.instance_info.add_static_atom(init.predicate, [arg for arg in init.args])
                else:
                    init_atom = self.instance_info.add_atom(init.predicate, [arg for arg in init.args])
                #self._print_atom(init_atom, init)
                init_atom_ids.append(init_atom.get_index())

        goal_atom_ids = []
        if isinstance(task.goal, pddl.conditions.Atom):
            goal_atom = self.instance_info.add_atom(f"{task.goal.predicate}_g", [arg for arg in task.goal.args])
            goal_atom_ids.append(goal_atom.get_index())
        else:
            for goal in task.goal.parts:
                goal_atom = self.instance_info.add_atom(f"{goal.predicate}_g", [arg for arg in goal.args])
                goal_atom_ids.append(goal_atom.get_index())

        #self.states = []
        self.current_state = State(set(init_atom_ids + goal_atom_ids), self.instance_info)
        #self.states.append(self.current_state)

        #print(self.current_state)

        self.actions = []

    def __bool__(self):
        return bool(self.actions)

    __nonzero__ = __bool__

    def _print_atom(self, dl_atom, pddl_atom, params_map=None):
        if params_map:
            print(f"\tAtom {dl_atom.get_index()}: {pddl_atom.predicate}({', '.join(params_map[arg] for arg in pddl_atom.args)})")
        else:
            print(f"\tAtom {dl_atom.get_index()}: {pddl_atom.predicate}({', '.join(arg for arg in pddl_atom.args)})")
    
    def _get_successor(self, state, action):
        succ_atom_ids = (state.atom_ids - action.get_del_atom_ids(self.instance_info)).union(action.get_add_atom_ids(self.instance_info))
        return State(succ_atom_ids, self.instance_info)

    def get_final_queue(self):
        return [a.action for a in self.actions if a.is_grounded]
    
    def print_info(self):
        print("Using PolicyQueue priority queue for actions.")
        
    def get_num_grounded_actions(self):
        return sum(a.is_grounded for a in self.actions)
    
    def get_num_actions(self):
        return len(self.actions)
    
    def push(self, action):
        self.actions.append(Action(action))
        if len(self.actions) == 100: # TODO play around with this
            self.sg = SuccessorGenerator(self.actions, self.instance_info)

        
    def pop(self):
        #print()
        #print(self.current_state.state)

        if self.sg:
            print(self.sg.get_applicable_actions(self.current_state))

        progress = True
        while progress:
            progress = False
            for action in reversed(self.actions):
                # TODO use successor generator
                if not self.current_state.is_applicable(action):
                    continue

                successor = self._get_successor(self.current_state, action)

                # TODO: to add the cache, we need proper state ids, self.eval_cache
                pol_eval = self.policy.evaluate_lazy(self.current_state.state, successor.state)

                if pol_eval:
                    #print(f"select new action {action.get_name()}")
                    #print(successor.state)
                    self.current_state = successor
                    if action.is_grounded:
                        progress = True
                        break
                    else:
                        action.is_grounded = True
                        return action.action

        assert False  # there should always be an action that satisfies the policy
