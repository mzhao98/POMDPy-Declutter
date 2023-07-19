from __future__ import print_function
from builtins import object
from pomdpy.discrete_pomdp import DiscreteAction


class ActionType(object):
    """
    Lists the possible actions and attributes an integer code to each for the object declutter sample problem
    """
    NOOP_NOOP = 0
    NOOP_BLUE = 1
    NOOP_RED = 2
    BLUE_NOOP = 3
    RED_NOOP = 4
    BLUE_RED = 5
    RED_BLUE = 6


class DeclutterAction(DiscreteAction):
    """
    -The Rock sample problem Action class
    -Wrapper for storing the bin number. Also stores the rock number for checking actions
    -Handles pretty printing
    """

    def __init__(self, bin_number):
        super(DeclutterAction, self).__init__(bin_number)


    def copy(self):
        return DeclutterAction(self.bin_number)

    def print_action(self):
        print(self.to_string())

    def to_string(self):
        action = str(self.bin_number)
        return action

    def distance_to(self, other_point):
        pass
