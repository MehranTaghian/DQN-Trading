from .Data import Data
import numpy as np


class DataAutoPatternExtractionAgent(Data):
    def __init__(self, data, state_mode, action_name, device, gamma, n_step=4, batch_size=50, window_size=1,
                 transaction_cost=0.0):
        """
        This data dedicates to non-sequential models. For this, we purely pass the observation space to the agent
        by candles or some representation of the candles. We even take a window of candles as input to such models
        despite being non-time-series to see how they perform on sequential data.
        :@param state_mode
                = 1 for OHLC
                = 2 for OHLC + trend
                = 3 for OHLC + trend + %body + %upper-shadow + %lower-shadow
                = 4 for %body + %upper-shadow + %lower-shadow
                = 5 a window of k candles + the trend of the candles inside the window
        :@param action_name
            Name of the column of the action which will be added to the data-frame of data after finding the strategy by
            a specific model.
        :@param device
            GPU or CPU selected by pytorch
        @param n_step: number of steps in the future to get reward.
        @param batch_size: create batches of observations of size batch_size
        @param window_size: the number of sequential candles that are selected to be in one observation
        @param transaction_cost: cost of the transaction which is applied in the reward function.
        """
        start_index_reward = 0 if state_mode != 5 else window_size - 1
        super().__init__(data, action_name, device, gamma, n_step, batch_size, start_index_reward=start_index_reward,
                         transaction_cost=transaction_cost)

        self.data_kind = 'AutoPatternExtraction'

        self.data_preprocessed = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm']].values
        self.state_mode = state_mode

        if state_mode == 1:  # OHLC
            self.state_size = 4

        elif state_mode == 2:  # OHLC + trend
            self.state_size = 5
            trend = self.data.loc[:, 'trend'].values[:, np.newaxis]
            self.data_preprocessed = np.concatenate([self.data_preprocessed, trend], axis=1)

        elif state_mode == 3:  # OHLC + trend + %body + %upper-shadow + %lower-shadow
            self.state_size = 8
            candle_data = self.data.loc[:, ['trend', '%body', '%upper-shadow', '%lower-shadow']].values
            self.data_preprocessed = np.concatenate([self.data_preprocessed, candle_data], axis=1)

        elif state_mode == 4:  # %body + %upper-shadow + %lower-shadow
            self.state_size = 3
            self.data_preprocessed = self.data.loc[:, ['%body', '%upper-shadow', '%lower-shadow']].values

        elif state_mode == 5:
            # window_size * OHLC
            self.state_size = window_size * 4
            temp_states = []
            for i, row in self.data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm']].iterrows():
                if i < window_size - 1:
                    temp_states += [row.open_norm, row.high_norm, row.low_norm, row.close_norm]
                else:
                    # The trend of the k'th index shows the trend of the whole candles inside the window
                    temp_states += [row.open_norm, row.high_norm, row.low_norm, row.close_norm]
                    self.states.append(np.array(temp_states))
                    # removing the trend and first 4 elements from the vector
                    temp_states = temp_states[3:-1]

        if state_mode < 5:
            for i in range(len(self.data_preprocessed)):
                self.states.append(self.data_preprocessed[i])

    def find_trend(self, window_size=20):
        self.data['MA'] = self.data.mean_candle.rolling(window_size).mean()
        self.data['trend_class'] = 0

        for index in range(len(self.data)):
            moving_average_history = []
            if index >= window_size:
                for i in range(index - window_size, index):
                    moving_average_history.append(self.data['MA'][i])
            difference_moving_average = 0
            for i in range(len(moving_average_history) - 1, 0, -1):
                difference_moving_average += (moving_average_history[i] - moving_average_history[i - 1])

            # trend = 1 means ascending, and trend = 0 means descending
            self.data['trend_class'][index] = 1 if (difference_moving_average / window_size) > 0 else 0
