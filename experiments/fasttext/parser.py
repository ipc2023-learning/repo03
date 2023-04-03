#! /usr/bin/env python

import math

from lab.parser import Parser

parser = Parser()
parser.add_pattern('num_ground_actions', 'Translator operators: (.+)', required=False, type=int)
parser.add_pattern('powerlifted_time', 'INFO     Successfully computed useful facts with Powerlifted and saved the outcome to useful_facts. \[(.+)s\]', required=False, type=float)
parser.add_pattern('grounding_queue_pushes', '(.+) total queue pushes', required=False, type=int)
parser.add_pattern('translator_time', 'Done! \[(.+)s CPU', required=False, type=float)


parser.parse()
