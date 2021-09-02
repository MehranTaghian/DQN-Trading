from .Data import Data
import torch
from sklearn.preprocessing import MinMaxScaler


class DataSequential(Data):
    def __init__(self, data, action_name, device, gamma, n_step=4, batch_size=50, window_size=20, transaction_cost=0.0):
        """
        This class is inherited from the Data class (which is the environment) and designed to change the state space
        to a sequence of candles in each time-step.
        @param data:
            which is of type DataLoader.data_train or DataLoader.data_test and is a data-frame
        @param action_name:
            Name of the column of the action which will be added to the data-frame of data after finding the strategy by
            a specific model.
        @param device: CPU or GPU
        @param n_step: number of steps in the future to get reward.
        @param batch_size: create batches of observations of size batch_size
        @param window_size: the number of sequential candles that are selected to be in one observation
        @param transaction_cost: cost of the transaction which is applied in the reward function.
        """
        super().__init__(data, action_name, device, gamma, n_step, batch_size, start_index_reward=(window_size - 1),
                         transaction_cost=transaction_cost)

        self.data_kind = 'LSTMSequential'
        self.state_size = 4

        self.data_preprocessed = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm']].values

        # We ignore the first window_size elements of the data because of trend
        # for i in range(window_size - 1, len(self.data_preprocessed) - window_size + 1):
        for i in range(0, len(self.data_preprocessed) - window_size + 1):
            temp_states = torch.zeros(window_size, self.state_size, device=device)
            for j in range(i, i + window_size):
                temp_states[j - i] = torch.tensor(
                    self.data_preprocessed[j], dtype=torch.float, device=device)

            self.states.append(temp_states.unsqueeze(1))

    def __next__(self):
        if self.index_batch < self.num_batch:
            batch = [s for s in
                     self.states[self.index_batch * self.batch_size: (self.index_batch + 1) * self.batch_size]]
            self.index_batch += 1
            return torch.cat(batch, dim=1)

        raise StopIteration
