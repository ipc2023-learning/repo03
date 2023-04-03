#! /usr/bin/env python

import math
import re

from lab.parser import Parser

parser = Parser()
parser.add_pattern('powerlifted_time', 'INFO     Successfully computed useful facts with Powerlifted and saved the outcome to useful_facts. \[(.+)s\]', required=False, type=float)


def parse_last_occurences(content, props):
    regexps = [["num_ground_actions", re.compile(r"Translator operators: (.+)"), False],
               ["grounding_queue_pushes", re.compile(r"(.+) total queue pushes"), False],
               ["translator_time", re.compile(r"Done! \[(.+)s CPU"), False],]
    for line in reversed(content):
        for i, entry in enumerate(regexps):
            name, regex, found = entry
            if not found:
                match = regex.search(line)
                if match:
                    if "time" in name:
                        props[name] = float(match.group(1))
                    else:
                        props[name] = int(match.group(1))
                    regexps[i][2] = True
        if (all(found for _, _, found in regexps)):
            break

parser.add_function(parse_last_occurences)

parser.parse()
