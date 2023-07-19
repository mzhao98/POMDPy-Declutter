from __future__ import print_function
from builtins import range
from pomdpy.discrete_pomdp import DiscreteState

BLUE = 0
GREEN = 1
RED = 2
YELLOW = 3
NO_ACTION = 4

def l2(loc1, loc2):
    return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

class CleanupState(DiscreteState):
    """
    The state contains the position of the robot, as well as a boolean value for each rock
    representing whether it is good (true => good, false => bad).

    This class also implements DiscretizedState in order to allow the state to be easily
    converted to a List
    """

    def __init__(self, color_states, human_pref):

        # self.color_states = color_states # [[# Blue taken, # blue total]....]
        # self.human_pref = human_pref  # list [Blue rew, Green rew, ...]
        #
        self.table_state = {'color_states': color_states,
                   "human_pref": human_pref}

    def flatten_to_1d(self, color_state):

        flat_state = [item for sublist in color_state for item in sublist]
        return flat_state

    def distance_to(self, other_table_state):
        """
        Distance is measured between beliefs by the sum of the num of different rocks
        """
        assert isinstance(other_table_state, TableState)
        distance = 0
        # distance = self.position.manhattan_distance(other_rock_state.position)
        for i, j in zip(self.flatten_to_1d(self.table_state['color_states']),
                        self.flatten_to_1d(other_table_state.table_state['color_states'])):
            if i != j:
                distance += 1

        for i, j in zip(self.table_state['human_pref'], other_rock_state.table_state['human_pref']):
            if i != j:
                distance += 1

        return distance

    def __eq__(self, other_table_state):
        return self.table_state == other_table_state.table_state

    def copy(self):
        return TableState(self.table_state['color_states'], self.table_state['human_pref'])

    def __hash__(self):
        """
        Returns a decimal value representing the binary state string
        :return:
        """
        return int(self.to_string(), 2)

    def to_string(self):
        color_state_str = ""
        color_state_str += f"Blue {BLUE}: {self.table_state['color_state'][BLUE][0]} picked out of {self.table_state['color_state'][BLUE][1]} total.\n"
        color_state_str += f"Green {GREEN}: {self.table_state['color_state'][GREEN][0]} picked out of {self.table_state['color_state'][GREEN][1]} total.\n"
        color_state_str += f"Red {RED}: {self.table_state['color_state'][RED][0]} picked out of {self.table_state['color_state'][RED][1]} total.\n"
        color_state_str += f"Yellow {YELLOW}: {self.table_state['color_state'][YELLOW][0]} picked out of {self.table_state['color_state'][YELLOW][1]} total.\n"

        color_state_str += "...Human Belief: \n"
        color_state_str += f"Blue {BLUE}: {self.table_state['human_pref'][BLUE]}\n"
        color_state_str += f"Blue {GREEN}: {self.table_state['human_pref'][GREEN]}\n"
        color_state_str += f"Blue {RED}: {self.table_state['human_pref'][RED]}\n"
        color_state_str += f"Blue {YELLOW}: {self.table_state['human_pref'][YELLOW]}\n"


        return color_state_str

    def print_state(self):
        """
        Pretty printing
        :return:
        """
        color_state_str = ""
        color_state_str += f"Blue {BLUE}: {self.table_state['color_state'][BLUE][0]} picked out of {self.table_state['color_state'][BLUE][1]} total.\n"
        color_state_str += f"Green {GREEN}: {self.table_state['color_state'][GREEN][0]} picked out of {self.table_state['color_state'][GREEN][1]} total.\n"
        color_state_str += f"Red {RED}: {self.table_state['color_state'][RED][0]} picked out of {self.table_state['color_state'][RED][1]} total.\n"
        color_state_str += f"Yellow {YELLOW}: {self.table_state['color_state'][YELLOW][0]} picked out of {self.table_state['color_state'][YELLOW][1]} total.\n"

        color_state_str += "...Human Belief: \n"
        color_state_str += f"Blue {BLUE}: {self.table_state['human_pref'][BLUE]}\n"
        color_state_str += f"Blue {GREEN}: {self.table_state['human_pref'][GREEN]}\n"
        color_state_str += f"Blue {RED}: {self.table_state['human_pref'][RED]}\n"
        color_state_str += f"Blue {YELLOW}: {self.table_state['human_pref'][YELLOW]}\n"
        print(f"State: {color_state_str} ")

    def as_list(self):
        """
        Returns a list containing the (i,j) grid position boolean values
        representing the boolean rock states (good, bad)
        :return:
        """
        pass

    def separate_rocks(self):
        """
        Used for the PyGame sim
        :return:
        """
        pass