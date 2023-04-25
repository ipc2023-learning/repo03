from collections import defaultdict

from enum import Enum
import io

import sys
import os
import string

sys.path.append(f'{os.path.dirname(__file__)}/../translate')
import pddl

class PredictionType(str, Enum):
    good_actions = 'good-actions'
    bad_actions = 'bad-actions'
    class_probability = 'class-probability'

class DeterminationType(str, Enum):
    all_out = 'all-out'
    all_out_except_one = 'all-out-except-one'

class BackgroundFileOptions:
    def __init__(self, add_negated_predicates, add_equal_predicate, use_object_types, prediction_type, determination_type):
        self.add_negated_predicates = add_negated_predicates
        self.add_equal_predicate = add_equal_predicate
        self.use_object_types = use_object_types
        self.prediction_type = prediction_type
        self.determination_type = determination_type

    def use_class_probability(self):
        return self.prediction_type==PredictionType.class_probability



# use_types is always false at the moment. The problem is that, in problems where there
# are object hierarchies, we need to have some form of "casting" that transforms the types
# above or below the hierarchy. This kind of defeats the purpose of using types because
# the overhead of casting could be larger han any savings by using types.
def get_type(type_name, use_plus, use_types):
    sign = "+" if use_plus else "-"
    typ = type_name if use_types else "object"
    return f"{sign}{get_type_name(typ)}"

def get_type_name(t):
    return f"'type:{t}'"

class BackgroundFile:

    def __init__(self, predicates, type_dict, opts):

        self.type_dict = type_dict
        self.aleph_base_file_content = ""

        self.opts = opts

        self.objects = defaultdict(set)
        # This could be very rare, but if we see the same object name with different types
        # in different tasks, we need to represent them with different names in the Aleph
        # configuration file. One way to do that is to change the prefix obj: by obj1: and
        # so on
        self.object_types_name = {}
        self.num_types_by_object = defaultdict(int)

        self.determination_backgrounds = []
        self.aleph_base_file_content = io.StringIO()
        self.aleph_fact_file_content = io.StringIO()

        if self.opts.use_class_probability():
            self.aleph_base_file_content.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
                                          "% specify tree type:\n"
                                          ":- set(tree_type,class_probability).\n"
                                          ":- set(classes,[ground,dont_ground]).\n"
                                          ":- set(dependent,DEPENDENT). % second arg of class is to predicted\n\n")

        self.predicates = [p for p in predicates if p.name != "="]

        self.aleph_base_file_content.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
                                      "% modes:\n")

        for predicate in self.predicates:
            arity = len(predicate.arguments)

            self.aleph_base_file_content.write(f":- discontiguous('ini:{predicate.name}'/{arity+1}).\n")
            self.aleph_base_file_content.write(f":- discontiguous('goal:{predicate.name}'/{arity+1}).\n")

            if opts.determination_type == DeterminationType.all_out_except_one:
                for i in range(arity):
                    params = "("
                    if (arity > 0):
                        params += ", ".join([get_type(predicate.arguments[j].type_name, i==j, opts.use_object_types) for j in range(arity)]) + ", "
                    params += "+task_id)"

                    self.aleph_base_file_content.write(":- modeb(*, 'ini:{predicate.name}'{params}).\n".format(**locals()))
                    self.aleph_base_file_content.write(":- modeb(*, 'goal:{predicate.name}'{params}).\n".format(**locals()))
                    if (self.opts.add_negated_predicates):
                        self.aleph_base_file_content.write(":- modeb(*, 'ini:not:{predicate.name}'{params}).\n".format(**locals()))
                        self.aleph_base_file_content.write(":- modeb(*, 'goal:not:{predicate.name}'{params}).\n".format(**locals()))

            else:

                params = "("
                if (arity > 0):
                    params += ", ".join([get_type(predicate.arguments[j].type_name, False, opts.use_object_types) for j in range(arity)]) + ", "
                params += "+task_id)"


                self.aleph_base_file_content.write(":- modeb(*, 'ini:{predicate.name}'{params}).\n".format(**locals()))
                self.aleph_base_file_content.write(":- modeb(*, 'goal:{predicate.name}'{params}).\n".format(**locals()))
                if (self.opts.add_negated_predicates):
                        self.aleph_base_file_content.write(":- modeb(*, 'ini:not:{predicate.name}'{params}).\n".format(**locals()))
                        self.aleph_base_file_content.write(":- modeb(*, 'goal:not:{predicate.name}'{params}).\n".format(**locals()))


            self.determination_backgrounds.append("'ini:{name}'/{size}".format(name = predicate.name, size = arity + 1))
            self.determination_backgrounds.append("'goal:{name}'/{size}".format(name = predicate.name, size = arity + 1))
            if (self.opts.add_negated_predicates):
                self.determination_backgrounds.append("'ini:not:{name}'/{size}".format(name = predicate.name, size = arity + 1))
                self.determination_backgrounds.append("'goal:not:{name}'/{size}".format(name = predicate.name, size = arity + 1))

        if (self.opts.add_equal_predicate):
            self.determination_backgrounds.append("equals/3")
            self.determination_backgrounds.append("nequals/3")
            self.aleph_base_file_content.write("\n")
            self.aleph_base_file_content.write(":- modeb(*, equals(+'type:object', +'type:object', +task_id)).\n")
            self.aleph_base_file_content.write(":- modeb(*, notequals(+'type:object', +'type:object', +task_id)).\n")

            self.aleph_base_file_content.write("notequals(A, B, Task):- obj(A, Task), obj(B, Task), not(equals(A, B, Task)).\n")

        self.aleph_base_file_content.write("\n")
        self.aleph_base_file_content.write("% types and constants\n")

        self.aleph_base_file_content.write("\n")



    def read_instance(self, task_run, task):
        object_names = {}
        for obj in task.objects:
            obj_type = obj.type_name if self.opts.use_object_types else 'object'

            self.objects[task_run].add(obj.name)
            if (obj.name, obj_type) in self.object_types_name:
                object_names[obj.name] = self.object_types_name[(obj.name, obj_type)]
            else:
                # We have never seen the same object with the same type
                modifier = self.num_types_by_object [obj.name] if self.num_types_by_object [obj.name] > 0 else ""
                object_names[obj.name] = f"'obj{modifier}:{obj.name}'"
                self.object_types_name[(obj.name, obj_type)]  = object_names[obj.name]

                self.num_types_by_object[obj.name] += 1


        bg_task_lines = [f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n% init {task_run}"]

        for ini_fact in task.init:
            if (type(ini_fact) == pddl.Assign or ini_fact.predicate == "="): # we have our own equality
                continue

            assert all ([arg in self.objects[task_run] for arg in ini_fact.args])

            bg_task_lines.append(f"'ini:{ini_fact.predicate}'(" + ",".join([f"{object_names[arg]}" for arg in ini_fact.args] + [f"'{task_run}')."]))


        bg_task_lines.append(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n% goal {task_run}")

        for goal_fact in task.goal.parts:
            assert all ([arg in self.objects[task_run] for arg in goal_fact.args])
            bg_task_lines.append(f"'goal:{goal_fact.predicate}'(" + ",".join([f"{object_names[arg]}" for arg in goal_fact.args] + [f"'{task_run}')."]))

        bg_task_lines.append("")

        self.aleph_fact_file_content.write("\n")

        self.aleph_fact_file_content.write("\n".join(bg_task_lines))


    def write(self, filename, class_parameters):
        with open(filename, "w") as b_file:
            arity = len(class_parameters)
            b_file.write(self.aleph_base_file_content.getvalue().replace("set(dependent,DEPENDENT)", "set(dependent,{D})".format(D=arity + (2 if self.opts.use_class_probability() else 1))))
            if (arity > 0):
                params = ", ".join([get_type(class_parameters[j], True, self.opts.use_object_types) for j in range(arity)]) + ", +task_id"
            else:
                params = "+task_id"


            if (self.opts.use_class_probability()):
                b_file.write(":- modeh(1, class({params}, -class)).\n\n".format(**locals()))
            else:
                b_file.write(":- modeh(1, class({params})).\n\n".format(**locals()))

            b_file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
            b_file.write("% determinations:\n")
            for bg in self.determination_backgrounds:
                b_file.write(":- determination(class/{arity}, {pred}).\n".format(arity=len(class_parameters) + (2 if self.opts.use_class_probability() else 1), pred=bg))

            b_file.write("\n")

            if self.opts.add_negated_predicates:
                b_file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
                b_file.write("% negated predicates:\n")
                for predicate in self.predicates:
                    if (len(predicate.arguments) > 26):
                        sys.exit("lazy programmer ERROR")
                    args = [x for x in string.ascii_uppercase[:len(predicate.arguments)]]
                    args_string = args[0]
                    for arg in args[1:]:
                        args_string += ", " + arg
                    b_file.write("'ini:not:{predicate.name}'({args_string}, Task):- ".format(**locals()))
                    b_file.write("obj({args[0]}, Task)".format(**locals()))
                    for arg in args[1:]:
                        b_file.write(", obj({arg}, Task)".format(**locals()))
                    b_file.write(", not('ini:{predicate.name}'({args_string}, Task)).\n".format(**locals()))
                    b_file.write("'goal:not:{predicate.name}'({args_string}, Task):- ".format(**locals()))
                    b_file.write("obj({args[0]}, Task)".format(**locals()))
                    for arg in args[1:]:
                        b_file.write(", obj({arg}, Task)".format(**locals()))
                    b_file.write(", not('goal:{predicate.name}'({args_string}, Task)).\n".format(**locals()))

                b_file.write("\n")

                for task in self.objects.keys():
                    b_file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
                    b_file.write("% all objects of task {task}:\n".format(**locals()))
                    for obj in self.objects[task]:
                        b_file.write("obj('obj:{obj}', '{task}').\n".format(**locals()))
                    b_file.write("\n")

            if self.opts.add_equal_predicate:
                for task in self.objects.keys():
                    b_file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
                    b_file.write("% equals task {task}:\n".format(**locals()))
                    for obj in self.objects[task]:
                        b_file.write("equals('obj:{obj}', 'obj:{obj}', '{task}').\n".format(**locals()))
                    b_file.write("\n")

                b_file.write("\n")


            type_declarations = defaultdict(list)
            for ((orig_name, object_type), object_name) in  self.object_types_name.items():
                queue = [object_type]
                explored = set()
                while queue:
                    object_type = queue.pop()
                    if object_type in explored:
                        continue
                    explored.add(object_type)
                    type_declarations[object_type].append(f"{get_type_name(object_type)}({object_name}).")
                    for supertype in self.type_dict[object_type].supertype_names:
                        queue.append(supertype)

            for t in type_declarations:
                b_file.write('\n'.join(type_declarations[t]))
                b_file.write('\n\n')


            b_file.write(self.aleph_fact_file_content.getvalue())

        # self.aleph_base_file_content.close()
        # self.aleph_fact_file_content.close()


def write_examples_file(filename, examples):
    with open(filename, "w") as n_file:
        n_file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        n_file.write("% training data:\n")
        for task in examples:
            for arguments in examples[task]:
                n_file.write(get_class_example(arguments, task))


def write_class_probability_examples_file(filename, good_examples, bad_examples, use_class_probability):
    with open(filename, "w") as f_file:
        f_file.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        f_file.write("% training data")

        for task in set(good_examples.keys()).union(set(bad_examples.keys())):
            for arguments in good_examples[task]:
                f_file.write("class('obj:{a}'".format(a=arguments[0]))
                for arg in arguments[1:]:
                    f_file.write(", 'obj:{arg}'".format(**locals()))
                if (use_class_probability):
                    f_file.write(", '{task}', ground).\n".format(**locals()))
                else:
                    f_file.write(", '{task}').\n".format(**locals()))

            for arguments in bad_examples[task]:
                f_file.write("class('obj:{a}'".format(a=arguments[0]))
                for arg in arguments[1:]:
                    f_file.write(", 'obj:{arg}'".format(**locals()))
                f_file.write(", '{task}', dont_ground).\n".format(**locals()))


def get_arg_list(action):
    return action.split("(")[1].replace(")", "").split(",")

def get_class_example(arguments, task):

    result = "class('obj:{a}'".format(a=arguments[0])
    for arg in arguments[1:]:
        result += ", 'obj:{arg}'".format(**locals())
    result += ", '{task}').\n".format(**locals())
    return result
