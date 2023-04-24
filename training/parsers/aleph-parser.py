#! /usr/bin/env python

from lab.parser import Parser
import os
import re
import json
from itertools import combinations

regex_summary = re.compile(r"\[Training set summary\] \[\[(\d+),(\d+),(\d+),(\d+)\]\]", re.MULTILINE)
regex_rule = re.compile(r'\[Pos cover = (\d+) Neg cover = (\d+)\]([^.]+)')
regex_rule_timeout = re.compile(r'(.+)\[pos cover = (\d+) neg cover = (\d+)\]')

class AlephParser(Parser):
    def __init__(self):
        Parser.__init__(self)

        with open('static-properties') as f:
            self.static_properties = json.load(f)


        self.add_pattern('accuracy', r"Accuracy = (.+)", type=float)
        self.add_pattern('total_time', r"\[time taken\] \[(.+)\]", type=float)

        self.add_function(self.has_theory)
        self.add_function(self.parse_aleph_log_file)

        for f in  os.listdir():
            if f.endswith('.h'):
                self.add_function(self.parse_aleph_hypothesis_file, file=f)

    def transform_hard_rule(self, rule, new_class_args):
        class_args = rule.split(":-")[0].split("(")[1].split(")")[0].split(",")[:-1]

        # If there are duplicate arguments in class args, we need to transform it into an equal rule
        new_rule_tuples = []
        for a, b in combinations(range(len(class_args)),2):
            if class_args[a] == class_args[b]:
                new_rule_tuples.append(f"equal:({new_class_args[a]},{new_class_args[b]})")

        if ":-" not in rule:
            if new_rule_tuples:
                return ";".join(new_rule_tuples).strip()
            else:
                return "true:"


        rule_tuples = rule[:-1].split(":-")[1].split(", ")# remove last argument, which is the task

        free_vars_args = {}
        num_free_vars_args = 0
        for (i, r) in enumerate(rule_tuples):
            r = ",".join(r.split(",")[:-1]).replace("'", "")
            predicates = r.split("),")
            for pred in predicates:
                if pred.startswith("("):
                    pred = pred[1:]
                if not pred:
                    continue

                pred_name, args = pred.replace(")", "").split("(")

                for arg in args.split(","):
                    if arg in class_args:
                        continue
                    if not arg in free_vars_args:
                        num_free_vars_args += 1
                        id_arg = num_free_vars_args
                        first_time = i
                    else:
                        (id_arg, first_time, _) = free_vars_args[arg]

                    free_vars_args[arg] = (id_arg, first_time, i)

        for i, r in enumerate(rule_tuples):
            r = ",".join(r.split(",")[:-1]).replace("'", "") +  ")" # remove last argument, which is the task
            predicates = r.split("),")
            for pred in predicates:
                if pred.startswith("("):
                    pred = pred[1:]
                if not pred:
                    continue
                pred_name, args = pred.replace(")", "").split("(")

                new_args = []
                for arg in args.split(","):
                    if arg in class_args:
                        new_args.append(new_class_args[class_args.index(arg)])
                    else:
                        (id_arg, first_time, last_time) = free_vars_args[arg]
                        if first_time == last_time:
                            name_arg = "_"
                        else:
                            name_arg = "?fv{}".format(id_arg)

                        new_args.append(name_arg)

                new_rule_tuples.append("{}({})".format(pred_name, ", ".join(new_args)))

        return ";".join(new_rule_tuples).strip()

    def has_theory(self, content, props):
        props['has_theory'] = '[theory]' in content

    def parse_aleph_hypothesis_file(self, content, props):
            lines = content.split('\n')

            rules = []
            # class_args = None
            for l in lines:
                if l.startswith("class"):
                    # class_args = l.split(":-")[0][6:].split(",")[:-1]
                    rules.append(l.strip())
                else:
                    rules[-1] += l.strip()

            props['rules.h'] = rules
            # props['class_args.h'] = class_args

    def parse_aleph_log_file(self, content, props):
        rules = []
        if props['has_theory']:
            content = content[content.index('[theory]'):]

            rules_text = content.split('[Rule ')[1:]
            for r in rules_text:
                r = r.replace('\n', ' ')
                match = regex_rule.search(r)
                rules.append((match[1], match[2], match[3].strip()))

            match = regex_summary.search(content)
            true_positives, false_positives, false_negatives, true_negatives = int(match[1]), int(match[2]), int(match[3]), int(match[4])

            props['true_positives'], props['false_positives'], props['false_negatives'], props['true_negatives'] = true_positives, false_positives, false_negatives, true_negatives
            props['precision'] = true_positives/(true_positives + false_positives)
            props['recall'] = true_positives/(true_positives + false_negatives)
            props['f_value'] = 2*props['precision']*props['recall']/(props['precision'] + props['recall'] )

        else:
            try:
                content = content[content.index('[best clause]') + len('[best clause]'):]
                relevant_text = content[:content.index(']')+1].replace('\n','')
                match = regex_rule_timeout.search(relevant_text)
                if int(match[2]) > 1: # Skip rules without at least 2 positive examples
                    rules.append((match[2], match[3], match[1].strip()))
            except:
                pass # No more best clause elements

        props['raw_rules'] = rules
        props['rules'] = []
        if rules:
            class_args = rules[0][2].split(":-")[0].split("(")[1].split(")")[0].split(",")[:-1]
            class_args = ["?arg{}".format(i) for i in range(len(class_args))]
            for _, _, rule in rules:
                props['rules'].append(self.static_properties['action_schema'] + " (" + ", ".join(class_args) + ")" + " :- " + self.transform_hard_rule (rule, class_args) + ".")


def main():
    parser = AlephParser()
    parser.parse()


main()
