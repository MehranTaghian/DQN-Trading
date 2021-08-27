import random
import numpy as np


def update_tuple(prev_tuple, index, value):
    temp = list(prev_tuple)
    temp[index] = value
    return tuple(temp)


class Agent:
    """
        :param Q_value is a dictionary containing the value of the state-action pairs. Each
        state is a dictionary of actions.
        Q_value[current_state][0]: value of the action = Buy Share in the current state.
        Q_value[current_state][1]: value of the action = Hold Share in the current state.
        Q_value[current_state][2]: value of the action = Sell Share in the current state.
        Q_value[current_state][3]: value of the action = No Share in the current state.

        This agent has only buy reward(no sell reward), available actions list, n-step sarsa,
        next-state action is included in the update part, action list contains 4 types of actions including
        buy share, hold share, sell share, no share.

        """

    def __init__(self, patterns, n=5, gamma=1, alpha=0.3, epsilon=0.01):
        """
        :param patterns: List of existing candle stick patterns to form states
        :param n: number of steps in SARSA algorithm
        :param gamma: for value iteration
        :param alpha: for value iteration
        :param epsilon: For ep-greedy algorithm
        """
        self.Q_value = {}
        self.pattern_to_code = {}
        self.code_to_pattern = {}
        for p in range(len(patterns)):
            self.pattern_to_code[patterns[p]] = p
            self.code_to_pattern[p] = patterns[p]

        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.own_share = False
        self.idle_state = tuple(np.zeros(len(patterns), dtype=int))
        self.n = n

    def value_iteration(self, data):
        # if self.convert_to_tuple(data.iloc[0]['label']) == self.idle_state:
        #     available_actions = [3]  # if idle, no share
        # else:
        available_actions = [0, 3]  # Buy Share or No Share

        for i in range(len(data) - 1):
            action, current_state = self.get_action_value(available_actions, data, i)
            available_actions = self.update_available_actions(current_state, action)
            action_prime, next_state = self.get_action_value(available_actions, data, i + 1)
            Q_prime = self.get_reward(action, data, i) + (self.gamma ** self.n) * self.Q_value[next_state][action_prime]
            Q = self.Q_value[current_state][action]
            self.Q_value[current_state][action] = (1 - self.alpha) * Q + self.alpha * Q_prime

    def get_action_value(self, available_actions, data, i):
        state = self.convert_to_tuple(data.iloc[i]['label'])
        self.check_state_in_state_value_dict(state)
        if len(available_actions) > 1:
            action = self.policy_epsilon_greedy(state, available_actions)
        else:
            action = available_actions[0]
        return action, state

    def take_action_with_policy(self, data):
        available_actions = [0, 1, 2, 3]
        for i in range(len(data) - 1):
            current_state = self.convert_to_tuple(data.iloc[i]['label'])
            self.check_state_in_state_value_dict(current_state)
            action = self.policy_epsilon_greedy(current_state, available_actions)
            available_actions = self.update_available_actions(current_state, action)
            yield action

    def policy_epsilon_greedy(self, state, available_actions):
        if random.random() <= self.epsilon:  # choose an action at random
            return available_actions[random.randint(0, len(available_actions) - 1)]
        else:  # choose the action with maximum value
            action_value_list = []
            for a in available_actions:
                action_value_list.append(self.Q_value[state][a])
            action_value_list = np.array(action_value_list)
            max_actions_indices = np.where(action_value_list == max(action_value_list))[0]
            if len(max_actions_indices) > 1:
                return available_actions[max_actions_indices[random.randint(0, len(max_actions_indices) - 1)]]
            return available_actions[max_actions_indices[0]]

    def get_reward(self, action, data, index):
        rewards = []
        self.calculate_raward_for_one_step(action, data, index, rewards)
        available_actions = self.update_available_actions(self.convert_to_tuple(data.iloc[index]['label']), action)

        for i in range(1, self.n + 1):
            if index + i < len(data):
                action_i, state_i = self.get_action_value(available_actions, data, index + i)
                self.calculate_raward_for_one_step(state_i, action_i, index + i, rewards)
                available_actions = self.update_available_actions(state_i, action_i)
            else:
                break

        reward = 0
        for i in range(len(rewards)):
            reward += (self.gamma ** i) * rewards[i]

        return reward

    def calculate_raward_for_one_step(self, action, data, index, rewards):
        if action == 0 or action == 1:  # Buy Share or Hold Share
            rewards.append(data.iloc[index + 1].close - data.iloc[index].close)
        elif action == 2 or action == 3:  # Sell Share or No Share
            rewards.append(0.0)

    def check_state_in_state_value_dict(self, current):
        if not (current in self.Q_value.keys()):
            self.Q_value[current] = {}
            self.Q_value[current][0] = 0.0
            self.Q_value[current][1] = 0.0
            self.Q_value[current][2] = 0.0
            self.Q_value[current][3] = 0.0

    def convert_to_tuple(self, labels):
        state = np.zeros(len(self.pattern_to_code), dtype=int)
        for l in labels:
            state[self.pattern_to_code[l]] = 1

        return tuple(state)

    def convert_to_label(self, tuple_state):
        pattern = []
        for i in tuple_state:
            if tuple_state[i] == 1:
                pattern.append(self.code_to_pattern[i])
        return pattern

    def check_idle(self, current_state, current_action):
        if current_state == self.idle_state:
            if current_action == 0 or current_action == 1:  # Buy or hold
                return 1  # State is Idle and action = hold share
            elif current_action == 2 or current_action == 3:  # Sell or No share
                return 3  # State is idle and action = No share
        return -1

    def update_available_actions(self, current_state, current_action):
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

    def count_number_of_idle_state(self, data):
        j = 0
        for i in range(len(data)):
            if self.convert_to_tuple(data.iloc[i]['label']) == self.idle_state:
                j += 1
        return j
