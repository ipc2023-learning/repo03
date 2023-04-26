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

        if 'class_probability' not in self.static_properties['config']:
            self.add_function(self.parse_aleph_log_file)


        if 'class_probability' in self.static_properties['config']:
            for f in  os.listdir():
                if f.endswith('.h'):
                    self.add_function(self.parse_class_probability_aleph_hypothesis_file, file=f)

    def transform_probability_class_rules(self, rules, class_args):
        rules_tuples = []
        for r in rules[::-1]:
            task_arg = r.split(":-")[0].split(",")[-2]
            #print ("Rule", r)
            if "not" in r:
                r = "".join(r.replace(":-not", ":-") [r.index(":-") + 2:].replace(",not", ", not").split(", not") [-1:]).replace(", random", ",random")
                predicate = r.split(",random")[0].replace("'", "").replace("," + task_arg, "").strip()
                ground_prob = [x for x in r.split(",") if "-ground" in x][0]
                #print ("Prob: ", ground_prob)
                ground_prob = float(ground_prob[:ground_prob.index("-ground")].replace("[", ""))
                rules_tuples.append((predicate, ground_prob))
            else:
                ground_prob = [x for x in r.split(",") if "-ground" in x][0]
                ground_prob = float(ground_prob[:ground_prob.index("-ground")].replace("[", ""))
                rules_tuples.append(("", ground_prob))


                #print(schema + "(" +  ",".join(class_args) +   ")  " + "; ".join(["{} {:f}".format(x[0], x[1]) for x in rules_tuples]))

        free_vars_args = {}
        num_free_vars_args = 0
        for (i, (r, g)) in enumerate(rules_tuples):
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

        new_rule_tuples = []
        for (i, (r, g)) in enumerate(rules_tuples):
            predicates = r.split("),")
            new_predicates = []
            for pred in predicates:
                if pred.startswith("("):
                    pred = pred[1:]
                if not pred:
                    continue
                pred_name, args = pred.replace(")", "").split("(")

                new_args = []
                for arg in args.split(","):
                    if arg in class_args:
                        new_args.append("?arg{}".format(class_args.index(arg)))
                    else:
                        (id_arg, first_time, last_time) = free_vars_args[arg]
                        if first_time == last_time:
                            name_arg = "_"
                        elif i == last_time:
                            name_arg = "?fv{}-i".format(id_arg)
                        elif i == first_time:
                            name_arg = "?fv{}-o".format(id_arg)
                        else:
                            name_arg = "?fv{}-io".format(id_arg)

                        new_args.append(name_arg)

                new_predicates.append("{}({})".format(pred_name, ",".join(new_args)))

            new_r = ",".join(new_predicates)
            new_rule_tuples.append((new_r, g))

        return new_rule_tuples

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

    def parse_class_probability_aleph_hypothesis_file(self, content, props):
            lines = content.split('\n')

            rules = []
            class_args = None
            for l in lines:
                if l.startswith("class"):
                    class_args = l.split(":-")[0][6:].split(",")[:-2]
                    rules.append(l.strip())
                else:
                    rules[-1] += l.strip()

            props['raw_rules'] = rules
            # props['class_args.h'] = class_args


            new_rule_tuples = self.transform_probability_class_rules(rules, class_args)
            props['class_probability_rule'] = self.static_properties['action_schema']  + " :- " + "; ".join(["{} {:f}".format(x[0], x[1]) for x in new_rule_tuples])


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
            props['precision'] = true_positives/(true_positives + false_positives) if true_positives else 0
            props['recall'] = true_positives/(true_positives + false_negatives) if true_positives else 0
            props['f_value'] = 2*props['precision']*props['recall']/(props['precision'] + props['recall'] ) if true_positives else 0

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
