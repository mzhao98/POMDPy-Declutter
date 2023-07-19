from __future__ import print_function
from builtins import object
from pomdpy.discrete_pomdp import DiscreteAction


class ActionType(object):
    """
    Enumerates the potential TigerActions
    """
    NOOP = 0
    BLUE = 1
    RED = 2


class TigerAction(DiscreteAction):
    def __init__(self, action_type):
        super(TigerAction, self).__init__(action_type)
        self.obj_selected = action_type

    def copy(self):
        return TigerAction(self.obj_selected)

    def to_string(self):
        if self.obj_selected is ActionType.NOOP:
            action = "no action"
        elif self.obj_selected is ActionType.BLUE:
            action = "blue"
        elif self.obj_selected is ActionType.RED:
            action = "red"
        else:
            action = "Unknown action type"
        return action

    def print_action(self):
        if self.obj_selected is ActionType.NOOP:
            print("no action")
        elif self.obj_selected is ActionType.BLUE:
            print("blue")
        elif self.obj_selected is ActionType.RED:
            print("red")
        else:
            print("Unknown action type")

    def distance_to(self, other_point):
        pass
