from .Data import Data
import torch
from sklearn.preprocessing import MinMaxScaler


class DataSequential(Data):
    def __init__(self, data, action_name, device, gamma, n_step=4, batch_size=50, window_size=20, transaction_cost=0.0):
        super().__init__(data, action_name, device, gamma, n_step, batch_size, start_index_reward=(window_size - 1),
                         transaction_cost=transaction_cost)
        self.data_kind = 'LSTMSequential'
        self.state_size = 4

        # self.find_trend(window_size)
        # self.data_preprocessed = data.loc[:, ['open', 'high', 'low', 'close', 'trend_sequential']].values

        self.data_preprocessed = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm']].values

        # scaler = MinMaxScaler()
        # scaler.fit(self.data_preprocessed)
        # self.data_preprocessed = scaler.transform(self.data_preprocessed)

        # We ignore the first window_size elements of the data because of trend
        # for i in range(window_size - 1, len(self.data_preprocessed) - window_size + 1):
        for i in range(0, len(self.data_preprocessed) - window_size + 1):
            temp_states = torch.zeros(window_size, self.state_size, device=device)
            for j in range(i, i + window_size):
                temp_states[j - i] = torch.tensor(
                    self.data_preprocessed[j], dtype=torch.float, device=device)

            self.states.append(temp_states.unsqueeze(1))

    def find_trend(self, window_size=20):
        self.data['MA'] = self.data.mean_candle.rolling(window_size).mean()
        self.data['trend_sequential'] = -10

        for index in range(len(self.data)):
            moving_average_history = []
            if index >= window_size - 1:
                for i in range(index - window_size + 1, index + 1):
                    moving_average_history.append(self.data['MA'][i])
            difference_moving_average = 0
            for i in range(len(moving_average_history) - 1, 0, -1):
                difference_moving_average += (moving_average_history[i] - moving_average_history[i - 1])

            # trend = 1 means ascending, and trend = 0 means descending
            self.data['trend_sequential'][index] = 1 if (difference_moving_average / window_size) > 0 else 0

    def __next__(self):
        if self.index_batch < self.num_batch:
            batch = [s for s in
                     self.states[self.index_batch * self.batch_size: (self.index_batch + 1) * self.batch_size]]
            self.index_batch += 1
            return torch.cat(batch, dim=1)

        raise StopIteration
