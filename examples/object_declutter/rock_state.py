from __future__ import print_function
from builtins import range
from pomdpy.discrete_pomdp import DiscreteState

BLUE = 1
RED = 2

color_to_text = {BLUE: "blue", RED: "red"}

class TableState(DiscreteState):
    """
    The state contains the set of objects remaining, as well as a value for whether the human
     has preference for blue or red (BLUE or RED).

    This class also implements DiscretizedState in order to allow the state to be easily
    converted to a List
    """

    def __init__(self, remaining_objects, human_pref):
        # if rock_states is not None:
        #     assert rock_states.__len__() is not 0
        self.remaining_objects = remaining_objects # list
        self.human_pref = human_pref  # int

    def distance_to(self, other_table_state):
        """
        Distance is measured between beliefs by the sum of the num of different objects
        and the difference in human preferences.
        """
        assert isinstance(other_table_state, TableState)
        distance = 0
        # distance = self.position.manhattan_distance(other_rock_state.position)
        for i, j in zip(self.remaining_objects, other_table_state.remaining_objects):
            if i != j:
                distance += 1
        if self.human_pref != other_table_state.human_pref:
            distance += 1
        return distance

    def __eq__(self, other_table_state):
        return self.remaining_objects == other_table_state.remaining_objects \
               and self.human_pref is other_table_state.human_pref

    def copy(self):
        return RockState(self.remaining_objects, self.human_pref)

    def __hash__(self):
        """
        Returns a decimal value representing the binary state string
        :return:
        """
        return int(self.to_string(), 2)

    def to_string(self):
        state_string = ""

        for obj in self.remaining_objects:
            state_string += str(obj) + " "

        state_string += " - " + str(self.human_pref)

        return state_string

    def print_state(self):
        """
        Pretty printing
        :return:
        """
        state_string = ""

        for obj in self.remaining_objects:
            state_string += color_to_text[obj] + " "

        state_string += " - human pref: " + color_to_text[self.human_pref]
        print(state_string)

    def as_list(self):
        """
        Returns a list containing the (i,j) grid position boolean values
        representing the boolean rock states (good, bad)
        :return:
        """
        state_list = []
        state_list.append(self.remaining_objects)
        state_list.append(self.human_pref)
        return state_list

