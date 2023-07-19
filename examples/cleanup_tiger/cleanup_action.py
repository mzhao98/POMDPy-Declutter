from __future__ import print_function
from builtins import object
from pomdpy.discrete_pomdp import DiscreteAction


class ActionType(object):
    """
    Enumerates the potential TigerActions
    """
    BLUE = 0
    GREEN = 1
    RED = 2
    YELLOW = 3
    NO_ACTION = 4


class CleanupAction(DiscreteAction):
    def __init__(self, action_type):
        super(CleanupAction, self).__init__(action_type)
        self.bin_number = action_type

    def copy(self):
        return CleanupAction(self.bin_number)

    def to_string(self):
        if self.bin_number is ActionType.BLUE:
            action = "BLUE"
        elif self.bin_number is ActionType.GREEN:
            action = "GREEN"
        elif self.bin_number is ActionType.RED:
            action = "RED"
        elif self.bin_number is ActionType.YELLOW:
            action = "YELLOW"
        elif self.bin_number is ActionType.NO_ACTION:
            action = "NO_ACTION"
        else:
            action = "Unknown action type"
        return action

    def print_action(self):
        if self.bin_number is ActionType.BLUE:
            action = "BLUE"
            print(f"Action {action}")
        elif self.bin_number is ActionType.GREEN:
            action = "GREEN"
            print(f"Action {action}")
        elif self.bin_number is ActionType.RED:
            action = "RED"
            print(f"Action {action}")
        elif self.bin_number is ActionType.YELLOW:
            action = "YELLOW"
            print(f"Action {action}")
        elif self.bin_number is ActionType.NO_ACTION:
            action = "NO_ACTION"
            print(f"Action {action}")
        else:
            print("Unknown action type")

    def distance_to(self, other_point):
        pass
