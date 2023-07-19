from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from builtins import str
from builtins import map
from builtins import hex
from builtins import range
from past.utils import old_div
import logging
import json
import numpy as np
from pomdpy.util import console, config_parser
# from .grid_position import GridPosition
from .rock_state import TableState
from .rock_action import DeclutterAction, ActionType
from .rock_observation import HumanActionObservation
from pomdpy.discrete_pomdp import DiscreteActionPool, DiscreteObservationPool
from pomdpy.pomdp import Model, StepResult
from .rock_position_history import HumanPrefData, HumanHistoryData

module = "RockModel"

BLUE = 1
RED = 2
NOOP = 0

class RockModel(Model):
    def __init__(self, args):
        super(RockModel, self).__init__(args)
        # logging utility
        self.logger = logging.getLogger('POMDPy.RockModel')
        # self.rock_config = json.load(open(config_parser.rock_cfg, "r"))

        # -------- Model configurations -------- #

        # The reward for sampling a good rock
        self.human_preference_reward = 2
        # The penalty for sampling a bad rock.
        self.human_dislike_obj_reward = -2
        # The reward for exiting the map
        self.task_completion_reward = 2
        # The penalty for any move.
        self.step_cost = 1 # cost must be subtracted



        self.initialize()

    # initialize the maps of the grid
    def initialize(self):
        self.possible_obj_states = [[BLUE, RED], [BLUE], [RED], []]
        self.human_pref_1 = BLUE # human preference for each object - Option 1
        self.human_pref_2 = RED # human preference for each object - Option 2
        self.possible_human_prefs = [self.human_pref_1, self.human_pref_2]

        self.possible_states = []
        for state in self.possible_obj_states:
            for pref in self.possible_human_prefs:
                self.possible_states.append((state, pref))

        # Total number of distinct states
        self.num_states = len(self.possible_states)
        self.possible_single_agent_actions = [BLUE, RED, NOOP]


        self.default_initial_state = ([BLUE, RED], self.human_pref_1)
        self.default_start_objs = [BLUE, RED]
        self.default_start_human_pref = self.human_pref_1

        legal_actions = []
        legal_actions.append(ActionType.NOOP_NOOP)
        legal_actions.append(ActionType.NOOP_BLUE)
        legal_actions.append(ActionType.NOOP_RED)
        legal_actions.append(ActionType.BLUE_NOOP)
        legal_actions.append(ActionType.RED_NOOP)
        legal_actions.append(ActionType.BLUE_RED)
        legal_actions.append(ActionType.RED_BLUE)
        self.possible_joint_actions = legal_actions
        self.num_actions = len(self.possible_joint_actions)

        self.remaining_objects = [BLUE, RED]
        self.num_times_human_took_blue = 0
        self.num_times_human_took_red = 0
        self.state_human_pref = self.default_start_human_pref

    ''' ===================================================================  '''
    '''                             Sampling                                 '''
    ''' ===================================================================  '''

    def sample_an_init_state(self):
        return TableState(self.default_start_objs, self.default_start_human_pref)


    def sample_state_informed(self, belief):
        return belief.sample_particle()

    def sample_state_uninformed(self):
        return TableState(self.default_start_objs, self.default_start_human_pref)

    ''' ===================================================================  '''
    '''                 Implementation of abstract Model class               '''
    ''' ===================================================================  '''

    def create_observation_pool(self, solver):
        return DiscreteObservationPool(solver)

    def is_terminal(self, env_state):
        # print("env_state", env_state)
        return env_state.remaining_objects == []

    def reset_for_simulation(self):
        self.remaining_objs = [BLUE, RED]
        self.num_times_human_took_blue = 0
        self.num_times_human_took_red = 0
        self.state_human_pref = self.default_start_human_pref


    def update(self, step_result):
        self.remaining_objects = step_result.next_state.remaining_objects
        if step_result.observation.human_pref == BLUE:
            self.num_times_human_took_blue += 1
        elif step_result.observation.human_pref == RED:
            self.num_times_human_took_red += 1

    def is_valid(self, state):
        return True

    def get_legal_actions(self, state):
        legal_actions = []
        # legal_actions = []
        legal_actions.append(ActionType.NOOP_NOOP)
        legal_actions.append(ActionType.NOOP_BLUE)
        legal_actions.append(ActionType.NOOP_RED)
        legal_actions.append(ActionType.BLUE_NOOP)
        legal_actions.append(ActionType.RED_NOOP)
        legal_actions.append(ActionType.BLUE_RED)
        legal_actions.append(ActionType.RED_BLUE)
        return legal_actions

    def get_max_undiscounted_return(self):

        return 3

    def reset_for_epoch(self):
        self.actual_human_pref = self.default_start_human_pref


    def get_all_states(self):
        """
        :return: Forgo returning all states to save memory, return the number of states as 2nd arg
        """
        return None, self.num_states

    def get_all_observations(self):
        """
        :return: Return a dictionary of all observations and the number of observations
        """
        return {
            "NOOP": 0,
            "BLUE": 1,
            "RED": 2
        }, 3

    def get_all_actions(self):
        """
        :return: Return a list of all actions along with the length
        """
        all_actions = self.possible_joint_actions
        return all_actions

    def create_action_pool(self):
        return DiscreteActionPool(self)

    def create_root_historical_data(self, solver):
        self.create_new_human_data()
        return HumanHistoryData(self, self.all_human_data, solver)

    def create_new_human_data(self):
        self.all_human_data = []
        self.all_human_data.append(HumanPrefData())

    def make_next_state(self, state, action):
        action_type = action.bin_number
        # next_position, is_legal = self.make_next_position(state.position.copy(), action_type)
        #
        # if not is_legal:
        #     # returns a copy of the current state
        #     return state.copy(), False

        remaining_objects = state.remaining_objects.copy()
        human_pref = state.human_pref

        # get action
        # NOOP_NOOP = 0
        # NOOP_BLUE = 1
        # NOOP_RED = 2
        # BLUE_NOOP = 3
        # RED_NOOP = 4
        # BLUE_RED = 5
        # RED_BLUE = 6
        human_action = NOOP
        robot_action = NOOP
        if action_type == ActionType.NOOP_BLUE:
            human_action = BLUE
        elif action_type == ActionType.NOOP_RED:
            human_action = RED
        elif action_type == ActionType.BLUE_NOOP:
            robot_action = BLUE
        elif action_type == ActionType.RED_NOOP:
            robot_action = RED
        elif action_type == ActionType.BLUE_RED:
            robot_action = BLUE
            human_action = RED
        elif action_type == ActionType.RED_BLUE:
            robot_action = RED
            human_action = BLUE

        # update remaining objects
        if human_action == BLUE and BLUE in remaining_objects:
            remaining_objects.remove(BLUE)
        elif human_action == RED and RED in remaining_objects:
            remaining_objects.remove(RED)

        if robot_action == BLUE and BLUE in remaining_objects:
            remaining_objects.remove(BLUE)
        elif robot_action == RED  and RED in remaining_objects:
            remaining_objects.remove(RED)

        return TableState(remaining_objects, human_pref), True

    def make_observation(self, initial_state, action, next_state):
        # generate new observation if not checking or sampling a rock
        human_pref, state_contains_human_pref = NOOP, False
        if human_pref in initial_state.remaining_objects:
            state_contains_human_pref = True
            human_pref = BLUE
        else:
            human_pref = NOOP

        return HumanActionObservation(human_pref, state_contains_human_pref)

    def belief_update(self, old_belief, action, observation):
        pass

    def make_reward(self, state, action, next_state, is_legal):

        if self.is_terminal(next_state):
            return self.task_completion_reward

        total_reward = -self.step_cost
        action_type = action.bin_number
        remaining_objects = state.remaining_objects.copy()
        human_pref = state.human_pref

        # get action
        # NOOP_NOOP = 0
        # NOOP_BLUE = 1
        # NOOP_RED = 2
        # BLUE_NOOP = 3
        # RED_NOOP = 4
        # BLUE_RED = 5
        # RED_BLUE = 6
        human_action = NOOP
        robot_action = NOOP
        if action_type == ActionType.NOOP_BLUE:
            human_action = BLUE
        elif action_type == ActionType.NOOP_RED:
            human_action = RED
        elif action_type == ActionType.BLUE_NOOP:
            robot_action = BLUE
        elif action_type == ActionType.RED_NOOP:
            robot_action = RED
        elif action_type == ActionType.BLUE_RED:
            robot_action = BLUE
            human_action = RED
        elif action_type == ActionType.RED_BLUE:
            robot_action = RED
            human_action = BLUE

        # print("human_action: ", human_action)
        # print("robot_action: ", robot_action)

        # update remaining objects
        if human_action == BLUE and BLUE in remaining_objects:
            remaining_objects.remove(BLUE)
            if self.actual_human_pref == BLUE:
                total_reward += self.human_preference_reward
            else:
                total_reward -= self.human_preference_reward
        elif human_action == RED and RED in remaining_objects:
            remaining_objects.remove(RED)
            if self.actual_human_pref == RED:
                total_reward += self.human_preference_reward
            else:
                total_reward -= self.human_preference_reward

        if robot_action == BLUE and BLUE in remaining_objects:
            remaining_objects.remove(BLUE)
            if self.actual_human_pref != BLUE:
                total_reward += self.human_preference_reward
            else:
                total_reward -= self.human_preference_reward
        elif robot_action == RED and RED in remaining_objects:
            remaining_objects.remove(RED)
            if self.actual_human_pref != RED:
                total_reward += self.human_preference_reward
            else:
                total_reward -= self.human_preference_reward

        if len(remaining_objects) == 0:
            total_reward += self.task_completion_reward
        return total_reward

    def generate_reward(self, state, action):
        next_state, is_legal = self.make_next_state(state, action)
        return self.make_reward(state, action, next_state, is_legal)

    def generate_step(self, state, action):
        if action is None:
            print("Tried to generate a step with a null action")
            return None
        elif type(action) is int:
            action = DeclutterAction(action)

        result = StepResult()
        result.next_state, is_legal = self.make_next_state(state, action)
        result.action = action.copy()
        result.observation = self.make_observation(state, action, result.next_state)
        result.reward = self.make_reward(state, action, result.next_state, is_legal)
        result.is_terminal = self.is_terminal(result.next_state)

        return result, is_legal

    # def generate_particles_uninformed(self, previous_belief, action, obs, n_particles):
    #     old_pos = previous_belief.get_states()[0].position
    #
    #     particles = []
    #     while particles.__len__() < n_particles:
    #         old_state = RockState(old_pos, self.sample_rocks())
    #         result, is_legal = self.generate_step(old_state, action)
    #         if obs == result.observation:
    #             particles.append(result.next_state)
    #     return particles

    def draw_env(self):
        print("Remaining Objects: " + str(self.remaining_objects))
        print("Human Pref: " + str(self.state_human_pref))
        return
