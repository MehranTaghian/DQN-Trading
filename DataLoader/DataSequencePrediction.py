from DataLoader.Data import Data
import torch
from SequencePredictor.TorchModel import LSTM
import os

from pathlib import Path


class DataSequencePrediction(Data):
    def __init__(self, data, action_name, model_file_name, device, gamma, n_step=4, batch_size=50, window_size=20):
        super().__init__(data, action_name, device, gamma, n_step, batch_size, start_index_reward=(window_size - 1))

        self.data_kind = 'SequencePrediction'

        self.state_size = 4

        file_path = os.path.join(Path(os.path.abspath(os.path.dirname(__file__))).parent,
                                 f'SequencePredictor\\Models\\{model_file_name}')

        self.model = LSTM(self.state_size, 32, self.state_size).to(device)
        self.model.load_state_dict(torch.load(file_path))
        self.model.eval()

        self.data_preprocessed = data.loc[:, ['open_norm', 'high_norm', 'low_norm', 'close_norm']].as_matrix()

        self.windowed_data = []
        # We ignore the first window_size elements of the data because of trend
        # for i in range(window_size - 1, len(self.data_preprocessed) - window_size + 1):
        for i in range(len(self.data_preprocessed) - window_size + 1):
            temp_states = torch.zeros(window_size, self.state_size, device=device)
            for j in range(i, i + window_size):
                temp_states[j - i] = torch.tensor(
                    self.data_preprocessed[j], dtype=torch.float, device=device)

            self.windowed_data.append(temp_states.unsqueeze(1))

        for i in range(len(self.windowed_data) - window_size):
            input_state = self.windowed_data[i]
            self.states.append(self.model(input_state).detach().unsqueeze(1))

    def __next__(self):
        if self.index_batch < self.num_batch:
            batch = [s for s in
                     self.states[self.index_batch * self.batch_size: (self.index_batch + 1) * self.batch_size]]
            self.index_batch += 1
            return torch.cat(batch, dim=1)

        raise StopIteration
