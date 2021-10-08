from .Data import Data
import numpy as np


class DataForPatternBasedAgent(Data):
    def __init__(self, data, patterns, action_name, device, gamma, n_step=4, batch_size=50, transaction_cost=0.0):
        """
        This class uses the original patterns extracted using the rules in the preprocessing step. You can find these
        rules in the ../PatternDetectionInCandleStick/LabelPatterns.py
        @param data:
            which is of type DataLoader.data_train or DataLoader.data_test and is a data-frame
        @param patterns:
            A dictionary holding different kinds of patterns.
        @param action_name:
            Name of the column of the action which will be added to the data-frame of data after finding the strategy by
            a specific model.
        @param device: CPU or GPU
        @param n_step: number of steps in the future to get reward.
        @param batch_size: create batches of observations of size batch_size
        """

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
