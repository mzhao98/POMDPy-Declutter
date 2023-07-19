from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from past.utils import old_div
import numpy as np
from pomdpy.pomdp import model
from .cleanup_action import *
from .cleanup_state import CleanupState
from .cleanup_observation import CleanupObservation
from .cleanup_data import CleanupData
from pomdpy.discrete_pomdp import DiscreteActionPool
from pomdpy.discrete_pomdp import DiscreteObservationPool
import copy

BLUE = 0
GREEN = 1
RED = 2
YELLOW = 3
NO_ACTION = 4
COLOR_LIST = [BLUE, GREEN, RED, YELLOW]
ACTION_LIST = [BLUE, GREEN, RED, YELLOW, NO_ACTION]


class CleanupModel(model.Model):
    def __init__(self, problem_name="Tiger"):
        super(CleanupModel, self).__init__(problem_name)
        self.human_pref = [0, 0, 0, 0]
        self.num_colors_total = [1, 1, 1, 1]
        # self.table_state = [[0,elem] for elem in self.num_colors_total]
        self.table_state = [0, 0, 0, 0]
        self.num_actions = 5

        self.state_list = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                      [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1],  [0, 0, 1, 1],
                      [1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1],  [0, 1, 1, 1],
                      [1, 1, 1, 1]]
        self.num_states = len(self.state_list)
        self.num_observations = 5

        self.possible_prefs = [[1, 1, -1, -1], [-1, -1, 1, 1]]

        # self.num_states = 2
        # self.num_observations = 2

    def start_scenario(self):
        self.human_pref = [-1, -0.5, 0.5, 1]


        # self.tiger_door = np.random.randint(0, self.num_doors) + 1

    ''' --------- Abstract Methods --------- '''

    # def get_num_remaining_objs(self, table_state):
    #     num_remain = 0
    #     for color_i in range(len(table_state)):
    #         num_remain += (table_state[color_i][1] - table_state[color_i][0])
    #     return num_remain

    def is_terminal(self, state):
        if sum(state) >= 4:
            return True
        else:
            return False

    def sample_an_init_state(self):
        return self.sample_state_uninformed()

    def create_observation_pool(self, solver):
        return DiscreteObservationPool(solver)

    def sample_state_uninformed(self):

        init_table_state = [0, 0, 0, 0]
        random_config = self.possible_prefs[1]
        if np.random.uniform(0, 1) <= 0.5:
            random_config = self.possible_prefs[0]
        return CleanupState(init_table_state, random_config)

    def sample_state_informed(self, belief):
        """

        :param belief:
        :return:
        """
        init_table_state = [0, 0, 0, 0]

        s = 100. * np.random.random()
        int1 = np.array([0., 100. * belief[0]])

        if int1[0] <= s <= int1[1]:
            return TigerState(init_table_state, self.possible_prefs[0])
        else:
            return TigerState(init_table_state, self.possible_prefs[1])

    # def flatten_to_tuple(self, state):
    #     # return tuple(list(sum(state, ())))
    #     return tuple([item for sublist in state for item in sublist])

    # def get_all_states(self):
    #     """
    #     Door is closed + either tiger is believed to be behind door 0 or door 1
    #     :return:
    #     """
    #     all_states = []
    #     table_state = [[0, elem] for elem in self.num_colors_total]
    #     # for a in COLOR_LIST:
    #     #     if table_state[a][1] - table_state[a][0] > 0:
    #     #         # is valid
    #     #         table_state[a][0] += 1
    #
    #
    #     visited_states = set()
    #     stack = [copy.deepcopy(table_state)]
    #     all_states.append(table_state)
    #
    #     while stack:
    #         state = stack.pop()
    #
    #         # convert old state to tuple
    #         state_tup = self.flatten_to_tuple(state)
    #
    #         # if state has not been visited, add it to the set of visited states
    #         if state_tup not in visited_states:
    #             visited_states.add(state_tup)
    #
    #         # get available
    #         if self.is_terminal(state):
    #             available_actions = []
    #         else:
    #             available_actions = COLOR_LIST
    #
    #         # get the neighbors of this state by looping through possible actions
    #         for idx, action in enumerate(available_actions):
    #             # set the environment to the current state
    #             table_state = copy.deepcopy(state)
    #             # take the action
    #             if table_state[action][1] - table_state[action][0] > 0:
    #                 # is valid
    #                 table_state[action][0] += 1
    #
    #
    #             new_state_tup = self.flatten_to_tuple(table_state)
    #
    #             if new_state_tup not in visited_states:
    #                 stack.append(copy.deepcopy(new_state))
    #
    #             if table_state not in all_states:
    #                 all_states.append(table_state)
    #
    #     final_all_states = []
    #     for human_pref in self.possible_prefs:
    #         for state in all_states:
    #             final_all_states.append((state, human_pref))
    #
    #     print("Total number of states", len(final_all_states))
    #     return final_all_states

    def get_all_states(self):
        all_states = []
        state_list = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                      [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1],  [0, 0, 1, 1],
                      [1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1],  [0, 1, 1, 1],
                      [1, 1, 1, 1]]
        for pref in self.possible_prefs:
            for state in state_list:
                all_states.append([state, pref])
        return all_states

    def get_all_actions(self):
        """
        Three unique actions
        :return:
        """
        return [CleanupAction(ActionType.BLUE), CleanupAction(ActionType.GREEN),
                CleanupAction(ActionType.RED)], CleanupAction(ActionType.YELLOW), CleanupAction(ActionType.NO_ACTION)

    def get_all_observations(self):
        """
        Either the roar of the tiger is heard coming from door 0 or door 1
        :return:
        """
        state_list = [0, 1, 2, 3, 4]
        return state_list

    def get_legal_actions(self, _):
        return self.get_all_actions()

    def is_valid(self, _):
        return True

    def reset_for_simulation(self):
        self.start_scenario()

    # Reset every "episode"
    def reset_for_epoch(self):
        self.start_scenario()

    def update(self, sim_data):
        pass

    def get_max_undiscounted_return(self):
        return 10

    @staticmethod
    def get_transition_matrix():
        """
        |A| x |S| x |S'| matrix, for tiger problem this is 3 x 2 x 2
        :return:
        """
        state_list = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                      [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1],
                      [1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1],
                      [1, 1, 1, 1]]
        possible_prefs = [[1, 1, -1, -1], [-1, -1, 1, 1]]
        all_states = []
        for pref in possible_prefs:
            for state in state_list:
                all_states.append([state, pref])

        all_obs = [0, 1, 2, 3, 4]
        all_actions = ACTION_LIST

        n_states = len(all_states)
        n_actions = len(all_actions)
        n_obs = len(all_obs)

        transition_mat = np.zeros((n_actions, n_states, n_states))

        for r_i in ACTION_LIST:
            r_action = ACTION_LIST[r_i]
            for i in range(len(all_states)):
                for i1 in range(len(all_states)):
                    state_i, pref_i = all_states[i][0], all_states[i][1]
                    state_i1, pref_i1 = all_states[i1][0], all_states[i1][1]

                    next_state = copy.deepcopy(state_i)
                    if r_action == NO_ACTION:
                        next_state = copy.deepcopy(state_i)
                    else:
                        if state_i[r_action] == 0:
                            # can take obj
                            next_state[r_action] = 1

                    h_action = NO_ACTION
                    max_rew = -1000
                    for color in COLOR_LIST:
                        if next_state[color] == 0 and pref_i[color] > max_rew:
                            max_rew = pref_i[color]
                            h_action = color


                    if h_action != NO_ACTION:
                        next_state[h_action] += 1

                    if next_state == state_i1 and pref_i == pref_i1:
                        transition_mat[r_i, i, i1] = 1.0
                    else:
                        transition_mat[r_i, i, i1] = 0.0
        return transition_mat


    @staticmethod
    def get_observation_matrix():
        """
        |A| x |S| x |O| matrix
        :return:
        """


        state_list = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                      [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1],  [0, 0, 1, 1],
                      [1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1],  [0, 1, 1, 1],
                      [1, 1, 1, 1]]
        possible_prefs = [[1, 1, -1, -1], [-1, -1, 1, 1]]
        all_states = []
        for pref in possible_prefs:
            for state in state_list:
                all_states.append([state, pref])

        all_obs = [0,1,2,3,4]
        all_actions = ACTION_LIST

        n_states = len(all_states)
        n_actions = len(all_actions)
        n_obs = len(all_obs)

        obs_mat = np.zeros((n_actions, n_states, n_obs))

        for r_i in ACTION_LIST:
            r_action = ACTION_LIST[r_i]
            for i in range(len(all_states)):
                for i1 in range(len(all_obs)):
                    state_i, pref_i = all_states[i][0], all_states[i][1]
                    obs_i1 = all_obs[i1]

                    next_state = copy.deepcopy(state_i)
                    if r_action == NO_ACTION:
                        next_state = copy.deepcopy(state_i)
                    else:
                        if state_i[r_action] == 0:
                            # can take obj
                            next_state[r_action] = 1

                    h_action = NO_ACTION
                    max_rew = -1000
                    for color in COLOR_LIST:
                        if next_state[color] == 0 and pref_i[color] > max_rew:
                            max_rew = pref_i[color]
                            h_action = color

                    # no_human_action_state = copy.deepcopy(next_state)
                    # if h_action != NO_ACTION:
                    #     next_state[h_action] += 1
                    #
                    #     if h_action == obs_i1:
                    #         obs_mat[r_i, i, i1] = 0.9
                    #
                    #     if h_action == obs_i1:
                    #         obs_mat[r_i, i, i1] = 0.1
                    #
                    # else:
                    if h_action == obs_i1:
                        obs_mat[r_i, i, i1] = 1.0
                    else:
                        obs_mat[r_i, i, i1] = 0.0

        return obs_mat

    @staticmethod
    def get_reward_matrix():
        """
        |A| x |S| matrix
        :return:
        """
        state_list = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                      [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1],
                      [1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1],
                      [1, 1, 1, 1]]
        possible_prefs = [[1, 1, -1, -1], [-1, -1, 1, 1]]
        all_states = []
        for pref in possible_prefs:
            for state in state_list:
                all_states.append([state, pref])

        all_obs = [0, 1, 2, 3, 4]
        all_actions = ACTION_LIST

        n_states = len(all_states)
        n_actions = len(all_actions)
        n_obs = len(all_obs)

        rew_mat = np.zeros((n_actions, n_states))

        for r_i in ACTION_LIST:
            r_action = ACTION_LIST[r_i]
            for i in range(len(all_states)):
                state_i, pref_i = all_states[i][0], all_states[i][1]

                next_state = copy.deepcopy(state_i)
                if r_action == NO_ACTION:
                    next_state = copy.deepcopy(state_i)
                else:
                    if state_i[r_action] == 0:
                        # can take obj
                        next_state[r_action] = 1
                        rew_mat[r_i, i] = 1.0
        return rew_mat

    def draw_env(self):
        print("Current state: ", self.table_state)
        print("Current preference: ", self.human_pref)

    @staticmethod
    def get_initial_belief_state():
        return np.array([0.5, 0.5])

    ''' Factory methods '''

    def create_action_pool(self):
        return DiscreteActionPool(self)

    def create_root_historical_data(self, agent):
        return CleanupData(self)

    ''' --------- BLACK BOX GENERATION --------- '''

    def generate_step(self, action, state=None):
        if action is None:
            print("ERROR: Tried to generate a step with a null action")
            return None
        elif not isinstance(action, CleanupAction):
            action = CleanupAction(action)

        result = model.StepResult()
        next_state_pair, obs, rew, done = self.make_step(action)
        result.is_terminal = done
        result.action = action.copy()
        result.observation = obs
        result.next_state = next_state_pair
        result.reward = rew

        return result


    def make_step(self, action):
        state_i, pref_i = self.table_state, self.human_pref

        rew = 0
        next_state = copy.deepcopy(state_i)
        if action == NO_ACTION:
            next_state = copy.deepcopy(state_i)
        else:
            if state_i[action] == 0:
                # can take obj
                next_state[action] = 1
                rew = 1

        h_action = NO_ACTION
        max_rew = -1000
        for color in COLOR_LIST:
            if next_state[color] == 0 and pref_i[color] > max_rew:
                max_rew = pref_i[color]
                h_action = color

        probability_correct = np.random.uniform(0, 1)
        if h_action != NO_ACTION and probability_correct <= 0.9:
            next_state[h_action] += 1

        obs = copy.deepcopy(next_state)
        obs = CleanupObservation(obs)

        next_state_pair = [next_state, pref_i]

        done = False
        if self.is_terminal(next_state):
            done = True

        return next_state_pair, obs, rew, done


    @staticmethod
    def make_next_state(action):
        if action.bin_number == ActionType.LISTEN:
            return False
        else:
            return True

    def make_reward(self, action, is_terminal):
        """
        :param action:
        :param is_terminal:
        :return: reward
        """

        if action.bin_number == ActionType.LISTEN:
            return -1.0

        if is_terminal:
            assert action.bin_number > 0
            if action.bin_number == self.tiger_door:
                ''' You chose the door with the tiger '''
                # return -20
                return -20.
            else:
                ''' You chose the door with the prize! '''
                return 10.0
        else:
            print("make_reward - Illegal action was used")
            return 0.0

    def make_observation(self, action):
        """
        :param action:
        :return:
        """
        if action.bin_number > 0:
            '''
            No new information is gained by opening a door
            Since this action leads to a terminal state, we don't care
            about the observation
            '''
            return TigerObservation(None)
        else:
            obs = ([0, 1], [1, 0])[self.tiger_door == 1]
            probability_correct = np.random.uniform(0, 1)
            if probability_correct <= 0.85:
                return TigerObservation(obs)
            else:
                obs.reverse()
                return TigerObservation(obs)

    def belief_update(self, old_belief, action, observation):
        """
        Belief is a 2-element array, with element in pos 0 signifying probability that the tiger is behind door 1

        :param old_belief:
        :param action:
        :param observation:
        :return:
        """
        # if action > 1:
        #     return old_belief

        p1_prior = old_belief[0]
        p2_prior = old_belief[1]

        h_action = observation.obs_state
        if h_action == NO_ACTION:
            likelihood_pref_1, likelihood_pref_2 = 0.5, 0.5
        else:
            likelihood_pref_1 = [elem - min(self.possible_prefs[0])  for elem in self.possible_prefs[0]]
            likelihood_pref_1 = [elem/sum(likelihood_pref_1) for elem in likelihood_pref_1]
            likelihood_pref_1 = likelihood_pref_1[h_action]

            likelihood_pref_2 = [elem - min(self.possible_prefs[1]) for elem in self.possible_prefs[1]]
            likelihood_pref_2 = [elem / sum(likelihood_pref_2) for elem in likelihood_pref_2]
            likelihood_pref_2 = likelihood_pref_2[h_action]

        # Observation 1 - the roar came from door 0
        observation_probability = (likelihood_pref_1 * p1_prior) + (likelihood_pref_2 * p2_prior)
        p1_posterior = old_div((likelihood_pref_1 * p1_prior),observation_probability)
        p2_posterior = old_div((likelihood_pref_2 * p2_prior),observation_probability)

        return np.array([p1_posterior, p2_posterior])
