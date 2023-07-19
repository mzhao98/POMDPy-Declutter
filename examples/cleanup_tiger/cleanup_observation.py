from __future__ import print_function
from pomdpy.discrete_pomdp import DiscreteObservation


class CleanupObservation(DiscreteObservation):
    """
    For num_doors = 2, there is an 85 % of hearing the roaring coming from the tiger door.
    There is a 15 % of hearing the roaring come from the reward door.

    source_of_roar[0] = 0 (door 1)
    source_of_roar[1] = 1 (door 2)
    or vice versa
    """

    def __init__(self, obs_state):
        super(CleanupObservation, self).__init__(obs_state)
        self.obs_state = obs_state

    def copy(self):
        return CleanupObservation(self.obs_state)

    def equals(self, other_observation):
        return self.obs_state == other_observation.obs_state

    def distance_to(self, other_observation):
        return (1, 0)[self.obs_state == other_observation.obs_state]

    def hash(self):
        return self.bin_number

    def print_observation(self):
        if self.obs_state is None:
            print("No observation from entering a terminal state")
        else:
            print(f"obs_state = {self.obs_state}")

    def to_string(self):
        obs = str(self.obs_state)
        return obs



