from __future__ import absolute_import
from __future__ import division
from builtins import str
from builtins import range
from past.utils import old_div
from builtins import object
import numpy as np
from pomdpy.pomdp import HistoricalData
from .rock_action import ActionType
import itertools
BLUE = 1
RED = 2
NOOP = 0

# Utility function
class HumanPrefData(object):
    """
    Stores data about each rock
    """

    def __init__(self):
        self.num_times_human_took_blue = 0
        # The calculated probability that the pref is BLUE
        self.chance_blue = 0.5

    def to_string(self):
        """
        Pretty printing
        """
        data_as_string = " Num BLUE took: " + \
                         str(self.num_times_human_took_blue) + \
                         " Probability that pref is blue: " + str(self.chance_blue)
        return data_as_string


class HumanHistoryData(HistoricalData):
    """
    A class to store the robot position associated with a given belief node, as well as
    explicitly calculated probabilities of goodness for each rock.
    """

    def __init__(self, model, all_human_data, solver):
        self.model = model
        self.solver = solver

        # List of RockData indexed by the rock number
        self.all_human_data = all_human_data

        # Holds reference to the function for generating legal actions
        if self.model.preferred_actions:
            self.legal_actions = self.generate_smart_actions
        else:
            self.legal_actions = self.generate_legal_actions

    @staticmethod
    def copy_human_data(other_data):
        new_human_data = []
        [new_human_data.append(HumanPrefData()) for _ in other_data]
        for i, j in zip(other_data, new_human_data):
            j.num_times_human_took_blue = i.num_times_human_took_blue
            j.chance_blue = i.chance_blue
        return new_human_data

    def copy(self):
        """
        Default behavior is to return a shallow copy
        """
        return self.shallow_copy()

    def deep_copy(self):
        """
        Passes along a reference to the rock data to the new copy of RockPositionHistory
        """
        return HumanHistoryData(self.model, self.all_human_data, self.solver)

    def shallow_copy(self):
        """
        Creates a copy of this object's rock data to pass along to the new copy
        """
        new_human_data = self.copy_human_data(self.all_human_data)
        return HumanHistoryData(self.model, new_human_data, self.solver)

    def update(self, other_belief):
        self.all_human_data = other_belief.data.all_human_data


    def create_child(self, joint_action, human_observation):
        next_data = self.deep_copy()
        # next_position, is_legal = self.model.make_next_position(self.grid_position.copy(), rock_action.bin_number)
        # next_data.grid_position = next_position

        # Update the human data
        probability_correct = 0.9
        probability_incorrect = 1 - probability_correct

        human_data = next_data.all_human_data[0]

        likelihood_blue = human_data.chance_blue
        likelihood_red = 1 - likelihood_blue

        if human_observation.human_pref == BLUE:
            human_data.num_times_human_took_blue += 1
            likelihood_blue *= probability_correct
            likelihood_red *= probability_incorrect
        elif human_observation.human_pref == RED:
            # human_data.num_times_human_took_blue += 1
            likelihood_blue *= probability_incorrect
            likelihood_red *= probability_correct

        if np.abs(likelihood_blue) < 0.01 and np.abs(likelihood_red) < 0.01:
            # No idea whether good or bad. reset data
            # print "Had to reset RockData"
            human_data = HumanPrefData()
        else:
            human_data.chance_blue = old_div(likelihood_blue, (likelihood_blue + likelihood_red))

        return next_data

    def generate_legal_actions(self):
        legal_actions = []
        # Add the following actions to legal actions list
        # NOOP_NOOP = 0
        # NOOP_BLUE = 1
        # NOOP_RED = 2
        # BLUE_NOOP = 3
        # RED_NOOP = 4
        # BLUE_RED = 5
        # RED_BLUE = 6

        legal_actions.append(ActionType.NOOP_NOOP)
        legal_actions.append(ActionType.NOOP_BLUE)
        legal_actions.append(ActionType.NOOP_RED)
        legal_actions.append(ActionType.BLUE_NOOP)
        legal_actions.append(ActionType.RED_NOOP)
        legal_actions.append(ActionType.BLUE_RED)
        legal_actions.append(ActionType.RED_BLUE)

        return legal_actions

    def generate_smart_actions(self):

        smart_actions = []


        human_data = self.all_human_data[0]
        if human_data.chance_blue > 0.5:
            smart_actions.append(NOOP_BLUE)
            smart_actions.append(RED_BLUE)
        elif human_data.chance_blue < 0.5:
            smart_actions.append(NOOP_RED)
            smart_actions.append(BLUE_RED)
        else:
            smart_actions.append(ActionType.NOOP_NOOP)
            smart_actions.append(ActionType.NOOP_BLUE)
            smart_actions.append(ActionType.NOOP_RED)
            smart_actions.append(ActionType.BLUE_NOOP)
            smart_actions.append(ActionType.RED_NOOP)
            smart_actions.append(ActionType.BLUE_RED)
            smart_actions.append(ActionType.RED_BLUE)

        return smart_actions







