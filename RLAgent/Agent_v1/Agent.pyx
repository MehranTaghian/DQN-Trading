import random
import numpy as np
cimport numpy as np

def update_tuple(prev_tuple, index, value):
    temp = list(prev_tuple)
    temp[index] = value
    return tuple(temp)

cdef class Agent:
    """
        :param Q_value is a dictionary containing the value of the state-action pairs. Each
        state is a dictionary of actions.
        Q_value[current_state][0]: value of the action = Buy Share in the current state.
        Q_value[current_state][1]: value of the action = Hold Share in the current state.
        Q_value[current_state][2]: value of the action = Sell Share in the current state.
        Q_value[current_state][3]: value of the action = No Share in the current state.

        This agent has only buy reward(no sell reward), available actions list, n-step sarsa,
        next-state action is included in the update part, action list contains 4 types of actions including
        buy share, hold share, sell share, no share. Here, we do not skip n-step when the action equals buy or sell,
        thus, we may sell at one time, and buy again in the next state (if the action-value of buy is max). Thus, we
        wrongly awarded the agent in the previous state.

        """
    cdef:
        public dict Q_value, pattern_to_code, code_to_pattern
        float gamma, alpha, epsilon
        int n
        tuple idle_state
        public list data_states, data_close_price

    def __init__(self, data, list patterns, int n=5, float gamma=1, float alpha=0.3, float epsilon=0.01):
        """
        :param patterns: List of existing candle stick patterns to form states
        :param n: number of steps in SARSA algorithm
        :param gamma: for value iteration
        :param alpha: for value iteration
        :param epsilon: For ep-greedy algorithm
        """
        self.pattern_to_code = {}
        self.code_to_pattern = {}

        for p in range(len(patterns)):
            self.pattern_to_code[patterns[p]] = p
            self.code_to_pattern[p] = patterns[p]

        self.data_states = []
        for i in data.label:
            self.data_states.append(self.convert_to_tuple(i))

        self.data_close_price = list(data.close)
        self.Q_value = {}

        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.idle_state = tuple(np.zeros(len(patterns), dtype=int))
        self.n = n

    cpdef void value_iteration(self):
        cdef:
            list available_actions = [0, 3]  # Buy Share or No Share
            int len_data = len(self.data_states)
            int i = 0
            int action, action_prime
            tuple current_state, next_state
            double Q, Q_prime

        while i < len_data - 1:
            action, current_state = self.find_optimal_state_action(available_actions, i)
            available_actions = self.update_available_actions(current_state, action)
            action_prime, next_state = self.find_optimal_state_action(available_actions, i + 1)
            Q_prime = self.get_reward(action, i) + (self.gamma ** self.n) * self.Q_value[next_state][action_prime]
            Q = self.Q_value[current_state][action]
            self.Q_value[current_state][action] = (1 - self.alpha) * Q + self.alpha * Q_prime

            i += 1

    cdef tuple find_optimal_state_action(self, list available_actions, int i):
        cdef:
            int action
            tuple state = self.data_states[i]

        self.check_state_in_state_value_dict(state)
        if len(available_actions) > 1:
            action = self.policy_epsilon_greedy(state, available_actions)
        else:
            action = available_actions[0]
        return action, state

    def take_action_with_policy(self, data):
        available_actions = [0, 1, 2, 3]
        for i in range(len(data) - 1):
            current_state = self.data_states[i]
            self.check_state_in_state_value_dict(current_state)
            action = self.policy_epsilon_greedy(current_state, available_actions)
            available_actions = self.update_available_actions(current_state, action)
            yield action

    cdef int policy_epsilon_greedy(self, tuple state, list available_actions):
        cdef:
            list action_value_list = []
            np.ndarray action_value_list_numpy
            np.ndarray max_actions_indices

        if random.random() <= self.epsilon:  # choose an action at random
            return available_actions[random.randint(0, len(available_actions) - 1)]
        else:  # choose the action with maximum value
            for a in available_actions:
                action_value_list.append(self.Q_value[state][a])
            action_value_list_numpy = np.array(action_value_list)
            max_actions_indices = np.where(action_value_list_numpy == max(action_value_list))[0]

            if len(max_actions_indices) > 1:
                return available_actions[max_actions_indices[random.randint(0, len(max_actions_indices) - 1)]]

            return available_actions[max_actions_indices[0]]

    cdef void calculate_reward_for_one_step(self, int action, int index, list rewards):
        if action == 0 or action == 1:  # Buy Share or Hold Share
            rewards.append(self.data_close_price[index + 1] - self.data_close_price[index])
        elif action == 2 or action == 3:  # Sell Share or No Share
            rewards.append(0.0)

    cdef double get_reward(self, int action, int index):
        cdef:
            list rewards = []
            list available_actions = []
            int i = 1
            int len_data = len(self.data_states)
            int action_i
            tuple state_i
            double reward = 0

        if index + 1 < len_data:
            self.calculate_reward_for_one_step(action, index, rewards)
            available_actions = self.update_available_actions(self.data_states[index], action)

        while i < self.n + 1:
            if index + i + 1 < len_data:
                action_i, state_i = self.find_optimal_state_action(available_actions, index + i)
                self.calculate_reward_for_one_step(action_i, index + i, rewards)
                available_actions = self.update_available_actions(state_i, action_i)
            else:
                break
            i += 1

        i = 0
        while i < len(rewards):
            reward += (self.gamma ** i) * rewards[i]
            i += 1
        return reward

    cdef void check_state_in_state_value_dict(self, tuple current):
        if not (current in self.Q_value.keys()):
            self.Q_value[current] = {}
            self.Q_value[current][0] = 0.0
            self.Q_value[current][1] = 0.0
            self.Q_value[current][2] = 0.0
            self.Q_value[current][3] = 0.0

    cdef tuple convert_to_tuple(self, list labels):
        cdef np.ndarray state = np.zeros(len(self.pattern_to_code), dtype=int)
        for l in labels:
            state[self.pattern_to_code[l]] = 1

        return tuple(state)

    def convert_to_label(self, tuple_state):
        pattern = []
        for i in range(len(tuple_state)):
            if tuple_state[i] == 1:
                pattern.append(self.code_to_pattern[i])
        if len(pattern) == 0:
            pattern.append('None')
        return pattern

    cdef int check_idle(self, tuple current_state, int current_action):
        if current_state == self.idle_state:
            if current_action == 0 or current_action == 1:  # Buy or hold
                return 1  # State is Idle and action = hold share
            elif current_action == 2 or current_action == 3:  # Sell or No share
                return 3  # State is idle and action = No share
        return -1

    cdef list update_available_actions(self, tuple current_state, int current_action):
        # action = self.check_idle(current_state, current_action)

        # if action == -1:
        if current_action == 0:  # Buy share
            return [1, 2]  # Sell or Hold share
        elif current_action == 1:  # Hold share
            return [1, 2]  # Hold or Sell share
        elif current_action == 2:  # Sell share
            return [0, 3]  # Buy or No share
        elif current_action == 3:  # No share
            return [0, 3]  # Buy or No share

        # return [action]

    cpdef count_number_of_idle_state(self, data):
        j = 0
        for i in range(len(data)):
            if self.convert_to_tuple(data.iloc[i]['label']) == self.idle_state:
                j += 1
        return j

    cpdef get_Qvalue_log(self):
        Q_value_log = {}
        for q in self.Q_value.keys():
            pattern_list = self.convert_to_label(q)
            label = str(pattern_list)
            if not (label in Q_value_log.keys()):
                Q_value_log[label] = {}

            Q_value_log[label]['buy'] = self.Q_value[q][0]
            Q_value_log[label]['hold'] = self.Q_value[q][1]
            Q_value_log[label]['sell'] = self.Q_value[q][2]
            Q_value_log[label]['no'] = self.Q_value[q][3]

            print(', '.join([str(i) for i in pattern_list]) + ' & ' + str(
                format(Q_value_log[label]['buy'], '.3f')) + ' & ' + str(
                format(Q_value_log[label]['hold'], '.3f')) + ' & ' +
                  str(format(Q_value_log[label]['sell'], '.3f')) + ' & ' + str(
                        format(Q_value_log[label]['no'], '.3f')) + '\\\\')

        return Q_value_log
