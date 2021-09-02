import random
import numpy as np
cimport numpy as np
cimport cython

def update_tuple(prev_tuple, index, value):
    temp = list(prev_tuple)
    temp[index] = value
    return tuple(temp)

cdef class Agent:
    """
        :@param Q_value is a dictionary containing the value of the state-action pairs. Each
        state is a dictionary of actions.
        Q_value[current_state][0]: value of the action = Buy Share in the current state.
        Q_value[current_state][1]: value of the action = None in the current state.
        Q_value[current_state][2]: value of the action = Sell Share in the current state.

        Here we updated the reward function, emit the next state value in updating action-value function,
        emit available actions list, if action is None, go one step, else we would give the reward of n step to the
        agent, thus, jump n steps (the same is true for selling). We also used None action for idle states (states with no patterns)
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
        :param n: number of steps in SARSA algorithm > 1
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
        """
        This function is the core of the training of our RL agent using n-step SARSA
        """
        cdef:
            list available_actions = [0, 1, 2]
            int len_data = len(self.data_states)
            int i = 0
            int action, action_prime
            tuple current_state, next_state
            double Q, Q_prime
            int own_share = False

        while i < len_data - self.n:
            current_state = self.data_states[i]
            available_actions = self.update_available_actions(current_state)
            action = self.find_optimal_state_action(current_state, available_actions)
            own_share = self.buy_or_sell_share(action, own_share)
            Q = self.Q_value[current_state][action]
            next_state = self.data_states[i + self.n]
            # set epsilon to False to find action greedy
            available_actions = self.update_available_actions(next_state)
            next_action = self.find_optimal_state_action(next_state, available_actions, False)
            Q_prime = self.Q_value[next_state][next_action]
            # self.Q_value[current_state][action] = (1 - self.alpha) * Q + \
            #                                       self.alpha * \
            #                                       (self.get_reward(action, i, own_share) +
            #                                        self.gamma ** self.n * Q_prime)
            self.Q_value[current_state][action] = (1 - self.alpha) * Q + self.alpha * self.get_reward(action, i,
                                                                                                      own_share)
            i += 1

    cdef int buy_or_sell_share(self, int action, int own_share):
        if action == 0:  # Buy
            return True
        elif action == 2:  # Sell
            return False
        return own_share

    cdef int find_optimal_state_action(self, state, list available_actions, use_epsilon=True):
        self.check_state_in_state_value_dict(state)
        if len(available_actions) > 1:
            return self.policy_epsilon_greedy(state, available_actions, use_epsilon)
        return available_actions[0]

    def use_new_data(self, data):
        self.data_states = []
        for i in data.label:
            self.data_states.append(self.convert_to_tuple(i))

        self.data_close_price = list(data.close)

    def take_action_with_policy(self, data):
        available_actions = [0, 1, 2]  # buy or None
        own_share = False
        self.use_new_data(data)
        for i in range(len(data) - 1):
            current_state = self.data_states[i]
            available_actions = self.update_available_actions(current_state)
            action = self.find_optimal_state_action(current_state, available_actions, False)
            own_share = self.buy_or_sell_share(action, own_share)
            yield action

    cdef int policy_epsilon_greedy(self, tuple state, list available_actions, use_epsilon=True):
        """
        returns an action which is either random (with probability epsilon) or the action that 
        has the maximum value in the Q-function
        @param state: current state
        @param available_actions: 
        @param use_epsilon: 
        @return: 
        """
        cdef:
            list action_value_list = []
            np.ndarray action_value_list_numpy
            np.ndarray max_actions_indices

        if use_epsilon and random.random() <= self.epsilon:  # choose an action at random
            return available_actions[random.randint(0, len(available_actions) - 1)]
        else:  # choose the action with maximum value
            for a in available_actions:
                action_value_list.append(self.Q_value[state][a])
            action_value_list_numpy = np.array(action_value_list)
            max_actions_indices = np.where(action_value_list_numpy == max(action_value_list))[0]

            if len(max_actions_indices) > 1:
                return available_actions[max_actions_indices[random.randint(0, len(max_actions_indices) - 1)]]

            return available_actions[max_actions_indices[0]]

    cdef void calculate_reward_for_one_step(self, int action, int index, list rewards, own_share):
        """
        The reward for selling is the opposite of the reward for buying, meaning that if some one sells his share and the
        value of the share increases, thus he should be punished. In addition, if some one sells appropriately and the value
        of the share decreases, he should be awarded
        :@param action: current action in the current state
        :@param index: index in the episode (each episode contains the whole candlesticks in the stock) 
        :@param rewards: rewards in the n-step reward
        :@param own_share: whether the agent is holding the stock or already sold it.
        """
        if action == 0 or (action == 1 and own_share):  # Buy Share or Hold Share
            rewards.append(self.data_close_price[index + 1] - self.data_close_price[index])
        elif action == 2 or (action == 1 and not own_share):  # Sell Share or No Share
            rewards.append(self.data_close_price[index] - self.data_close_price[index + 1])  # opposite of the buy case

    cdef double get_reward(self, int action, int index, own_share):
        cdef:
            list rewards = []
            # We see a state after the close price, so we start rewarding the day after, because the agent
            # would buy the stock in the next day. So i = 1
            int i = 1
            int len_data = len(self.data_states)
            int action_i
            tuple state_i
            double reward = 0

        while i < self.n:
            if index + i + 1 < len_data:
                self.calculate_reward_for_one_step(action, index + i, rewards, own_share)
            else:
                break
            i += 1

        i = 0
        while i < len(rewards):
            reward += (self.gamma ** i) * rewards[i]
            i += 1
        return reward

    cdef void check_state_in_state_value_dict(self, tuple current):
        """
        If the state was first visited, it will create an entry in the Q-table
        """
        if not (current in self.Q_value.keys()):
            self.Q_value[current] = {}
            self.Q_value[current][0] = 0.0
            self.Q_value[current][1] = 0.0
            self.Q_value[current][2] = 0.0

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

    cdef list update_available_actions(self, tuple next_state):
        if next_state == self.idle_state:
            return [1]
        else:
            return [0, 1, 2]

    cpdef count_number_of_idle_state(self, data):
        j = 0
        for i in range(len(data)):
            if self.convert_to_tuple(data.iloc[i]['label']) == self.idle_state:
                j += 1
        return j