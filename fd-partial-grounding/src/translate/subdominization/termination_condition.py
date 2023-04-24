#! /usr/bin/env python
# -*- coding: utf-8 -*-

import options
import pddl

import sys


class TerminationCondition:
    def print_info(self):
        pass

    def terminate(self):
        pass

    def notify_atom(self, atom):
        pass


class DefaultCondition(TerminationCondition):
    def print_info(self):
        print("Using default termination condition, i.e. grounding all actions.")

    def terminate(self):
        return False

    def notify_atom(self, atom):
        pass


class GoalRelaxedReachableCondition(TerminationCondition):
    def __init__(self):
        self.goal_reached = False
    def print_info(self):
        print("Grounding stopped if goal is relaxed reachable.")

    def terminate(self):
        return self.goal_reached

    def notify_atom(self, atom):
        if not self.goal_reached and isinstance(atom.predicate, str) and atom.predicate == "@goal-reachable":
            self.goal_reached = True


class GoalRelaxedReachablePlusNumberCondition(TerminationCondition):
    def __init__(self, num_additional_actions):
        self.goal_reached = False
        self.num_additional_actions = num_additional_actions
    def print_info(self):
        print("Grounding stopped if goal is relaxed reachable + %d additional actions." % self.num_additional_actions)

    def terminate(self):
        return self.goal_reached and self.num_additional_actions <= 0

    def notify_atom(self, atom):
        if self.goal_reached:
            if isinstance(atom.predicate, pddl.Action):
                self.num_additional_actions -= 1
        elif isinstance(atom.predicate, str) and atom.predicate == "@goal-reachable":
            self.goal_reached = True


class GoalRelaxedReachableMinNumberCondition(TerminationCondition):
    def __init__(self, min_num_actions):
        self.goal_reached = False
        self.min_num_actions = min_num_actions

    def print_info(self):
        print("Grounding stopped if goal is relaxed reachable and at least %d actions have been grounded." % self.min_num_actions)

    def terminate(self):
        return self.goal_reached and self.min_num_actions <= 0

    def notify_atom(self, atom):
        if isinstance(atom.predicate, pddl.Action):
            self.min_num_actions -= 1
        elif isinstance(atom.predicate, str) and atom.predicate == "@goal-reachable":
            self.goal_reached = True


class GoalRelaxedReachablePlusPercentageCondition(TerminationCondition):
    def __init__(self, percentage_additional_actions):
        self.goal_reached = False
        self.percentage_additional_actions = percentage_additional_actions
        if self.percentage_additional_actions < 0:
            exit("ERROR: percentage of additional actions must be >=0")
        self.number_grounded_actions = 0

    def print_info(self):
        print("Grounding stopped if goal is relaxed reachable + "
              f"{self.percentage_additional_actions}% additional actions.")

    def terminate(self):
        return self.goal_reached and self.num_additional_actions <= 0

    def notify_atom(self, atom):
        if self.goal_reached:
            if isinstance(atom.predicate, pddl.Action):
                self.num_additional_actions -= 1
        elif isinstance(atom.predicate, str) and atom.predicate == "@goal-reachable":
            self.goal_reached = True
            self.num_additional_actions = self.number_grounded_actions * self.percentage_additional_actions / 100
        elif isinstance(atom.predicate, pddl.Action):
            self.number_grounded_actions += 1


class GoalRelaxedReachablePlusPercentageMinIncrementCondition(TerminationCondition):
    def __init__(self, percentage_additional_actions, min_increment):
        self.goal_reached = False
        self.percentage_additional_actions = percentage_additional_actions
        if self.percentage_additional_actions < 0:
            exit("ERROR: percentage of additional actions must be >=0")
        self.min_increment = min_increment
        if self.min_increment < 0:
            exit("ERROR: minimum increment must be >=0")
        self.number_grounded_actions = 0

    def print_info(self):
        print("Grounding stopped if goal is relaxed reachable + at least "
              "max(#RS * {percentage}, {min_inc}) + #RS actions "
              "have been grounded, where \"#RS\" is the number of actions grounded "
              "when the goal becomes relaxed reachable.".format( 
                        percentage = (100 + self.percentage_additional_actions) / 100,
                        min_inc = self.min_increment))

    def terminate(self):
        return self.goal_reached and self.num_additional_actions <= 0

    def notify_atom(self, atom):
        if (self.goal_reached):
            if (isinstance(atom.predicate, pddl.Action)):
                self.num_additional_actions -= 1
        elif (isinstance(atom.predicate, str) and atom.predicate == "@goal-reachable"):
            self.goal_reached = True
            self.num_additional_actions = max(self.number_grounded_actions * self.percentage_additional_actions / 100, 
                                              self.min_increment)
        elif (isinstance(atom.predicate, pddl.Action)):
            self.number_grounded_actions += 1


class GoalRelaxedReachableMinNumberPlusPercentageMaxIncrementCondition(TerminationCondition):
    def __init__(self, min_number_actions, percentage_additional_actions, max_increment):
        self.goal_reached = False
        self.percentage_additional_actions = percentage_additional_actions
        if (self.percentage_additional_actions < 0):
            exit("ERROR: percentage of additional actions must be >=0")
        self.min_number_actions = min_number_actions
        if (self.min_number_actions < 0):
            exit("ERROR: minimum number of actions must be >=0")
        self.max_increment = max_increment
        if (self.max_increment < 0):
            exit("ERROR: maximum increment must be >=0")
        self.number_grounded_actions = 0

    def print_info(self):
        print("Grounding stopped if goal is relaxed reachable + at least "
              "max({min_number}, min(#RS * {percentage}, {max_inc}) + #RS) actions "
              "have been grounded, where \"#RS\" is the number of actions grounded "
              "when the goal becomes relaxed reachable.".format(min_number = self.min_number_actions, 
                        percentage = (100 + self.percentage_additional_actions) / 100,
                        max_inc = self.max_increment))

    def terminate(self):
        return self.goal_reached and self.num_additional_actions <= 0

    def notify_atom(self, atom):
        if (self.goal_reached):
            if (isinstance(atom.predicate, pddl.Action)):
                self.num_additional_actions -= 1
        elif (isinstance(atom.predicate, str) and atom.predicate == "@goal-reachable"):
            self.goal_reached = True
            self.num_additional_actions = max(self.min_number_actions - self.number_grounded_actions, 
                                              min(self.number_grounded_actions * self.percentage_additional_actions / 100, 
                                                  self.max_increment))
        elif (isinstance(atom.predicate, pddl.Action)):
            self.number_grounded_actions += 1


def get_termination_condition_from_options():
    args = options.termination_condition
    if (len(args) == 1):
        if (args[0] == "default"):
            return DefaultCondition()
        elif (args[0] == "goal-relaxed-reachable"):
            return GoalRelaxedReachableCondition()
        else:
            sys.exit("Error: unknown termination condition: " + args[0])
    elif (len(args) == 3):
        if (args[0] == "goal-relaxed-reachable"):
            if (args[1] == "number"):
                return GoalRelaxedReachablePlusNumberCondition(int(args[2]))
            if (args[1] == "min-number"):
                return GoalRelaxedReachableMinNumberCondition(int(args[2]))
            elif (args[1] == "percentage"):
                return GoalRelaxedReachablePlusPercentageCondition(int(args[2]))
            else:
                sys.exit("ERROR: unknown option for goal-relaxed-reachable termination condition " + args[1])
        else:
            sys.exit("ERROR: unknown termination condition " + args[0])
    elif (len(args) == 5):
        if (args[0] == "goal-relaxed-reachable"):
            percentage = -1
            min_increment = -1
            i = 1
            while (i < len(args)):
                if (args[i] == "percentage"):
                    percentage = int(args[i+1])
                    i += 2
                elif (args[i] == "min-increment"):
                    min_increment = int(args[i+1])
                    i += 2
                else:
                    sys.exit("ERROR: unknown option for termination condition " + args[i])
            return GoalRelaxedReachablePlusPercentageMinIncrementCondition(percentage, min_increment)
        else:
            sys.exit("ERROR: unknown termination condition " + args[0])
    elif len(args) == 7:
        if (args[0] == "goal-relaxed-reachable"):
            min_number = -1
            percentage = -1
            max_increment = -1
            i = 1
            while (i < len(args)):
                if (args[i] == "min-number"):
                    min_number = int(args[i+1])
                    i += 2
                elif (args[i] == "percentage"):
                    percentage = int(args[i+1])
                    i += 2
                elif (args[i] == "max-increment"):
                    max_increment = int(args[i+1])
                    i += 2
                else:
                    exit("ERROR: unknown option for termination condition " + args[i])
            return GoalRelaxedReachableMinNumberPlusPercentageMaxIncrementCondition(min_number, percentage, max_increment)
        else:
            exit("ERROR: unknown termination condition " + args[0])
    else:
        exit("ERROR: unrecognized termination condition " + str(args))
        
        
