from __future__ import absolute_import
from pomdpy.pomdp import HistoricalData
from .cleanup_action import ActionType
import numpy as np


class CleanupData(HistoricalData):
    """
    Used to store the probabilities that the tiger is behind a certain door.
    This is the belief distribution over the set of possible states.
    For a 2-door system, you have
        P( X = 0 ) = p
        P( X = 1 ) = 1 - p
    """
    def __init__(self, model):
        self.model = model
        self.pickup_count = 0
        ''' Initially there is an equal probability of the tiger being in either door'''
        self.pref_probabilities = [0.5, 0.5]
        self.legal_actions = self.generate_legal_actions

    def copy(self):
        dat = CleanupData(self.model)
        dat.listen_count = self.listen_count
        dat.pref_probabilities = self.pref_probabilities
        return dat

    def update(self, other_belief):
        self.pref_probabilities = other_belief.data.pref_probabilities

    def create_child(self, action, observation):
        next_data = self.copy()

        if action.bin_number > 1:
            ''' for open door actions, the belief distribution over possible states isn't changed '''
            return next_data
        else:
            self.pickup_count += 1
            '''
            Based on the observation, the door probabilities should change here.
            This is the key update that affects value function
            '''

            ''' ------- Bayes update of belief state -------- '''

            next_data.pref_probabilities = self.model.belief_update(np.array([self.pref_probabilities]), action,
                                                                    observation)
        return next_data

    @staticmethod
    def generate_legal_actions():
        """
        At each non-terminal state, the agent can listen or choose to open the door based on the current door probabilities
        :return:
        """
        return [ActionType.BLUE, ActionType.GREEN, ActionType.RED, ActionType.YELLOW, ActionType.NO_ACTION]

