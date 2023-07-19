from __future__ import print_function
from builtins import str
from pomdpy.discrete_pomdp import DiscreteObservation

BLUE = 1
RED = 2
color_to_text = {BLUE: "blue", RED: "red"}

class HumanActionObservation(DiscreteObservation):
    """
    Default behavior is for the rock observation to say that the rock is empty
    """
    def __init__(self, human_pref, state_contains_human_pref):
        super(HumanActionObservation, self).__init__(human_pref)
        self.human_pref = human_pref
        self.state_contains_human_pref = state_contains_human_pref

    def distance_to(self, other_human_action_observation):
        return abs(self.human_pref - other_human_action_observation.human_pref) + \
               abs(self.state_contains_human_pref - other_human_action_observation.state_contains_human_pref)

    def copy(self):
        return RockObservation(self.human_pref, self.state_contains_human_pref)

    def __eq__(self, other_rock_observation):
        return self.human_pref == other_rock_observation.human_pref and \
                  self.state_contains_human_pref == other_rock_observation.state_contains_human_pref

    def __hash__(self):
        return self.human_pref

    def print_observation(self):
        print("Human Pref: " + str(self.human_pref) +
              " State Contains Human Pref: " + str(self.state_contains_human_pref))

    def to_string(self):
        obs = str(self.human_pref) + " " + str(self.state_contains_human_pref)
        return obs
