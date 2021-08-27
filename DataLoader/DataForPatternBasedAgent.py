from .Data import Data
import numpy as np


class DataForPatternBasedAgent(Data):
    def __init__(self, data, patterns, action_name, device, gamma, n_step=4, batch_size=50, transaction_cost=0.0):
        super().__init__(data, action_name, device, gamma, n_step, batch_size, transaction_cost=transaction_cost)

        self.data_kind = 'PatternBased'

        self.pattern_to_code = {}
        self.code_to_pattern = {}

        self.state_size = len(patterns)
        self.idle_state = np.zeros(len(patterns))

        for p in range(len(patterns)):
            self.pattern_to_code[patterns[p]] = p
            self.code_to_pattern[p] = patterns[p]

        for i in data.label:
            self.states.append(self.convert_to_tuple(i))

    def convert_to_tuple(self, labels):
        state = np.zeros(len(self.pattern_to_code), dtype=int)
        for l in labels:
            state[self.pattern_to_code[l]] = 1

        return state
