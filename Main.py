# Importing DataLoaders for each model. These models include rule-based, vanilla DQN and encoder-decoder DQN.
from DataLoader.DataLoader import YahooFinanceDataLoader
from DataLoader.DataForPatternBasedAgent import DataForPatternBasedAgent
from DataLoader.DataAutoPatternExtractionAgent import DataAutoPatternExtractionAgent
from DataLoader.DataSequential import DataSequential

from DeepRLAgent.MLPEncoder.Train import Train as SimpleMLP
from DeepRLAgent.SimpleCNNEncoder.Train import Train as SimpleCNN
from EncoderDecoderAgent.GRU.Train import Train as GRU
from EncoderDecoderAgent.CNN.Train import Train as CNN
from EncoderDecoderAgent.CNN2D.Train import Train as CNN2d
from EncoderDecoderAgent.CNNAttn.Train import Train as CNN_ATTN
from EncoderDecoderAgent.CNN_GRU.Train import Train as CNN_GRU

# Imports for Deep RL Agent
from DeepRLAgent.VanillaInput.Train import Train as DeepRL

import torch
import argparse

parser = argparse.ArgumentParser(description='DQN-Trader arguments')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--models', help="Enter the name of the models you want to do HP sensitivity. The names should be"
                                     "separated using space.E.g. 'MLP DRL CNN'")
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()


class SensitivityRun:
    def __init__(self,
                 data_loader,
                 dataset_name,
                 gamma,
                 batch_size,
                 replay_memory_size,
                 feature_size,
                 target_update,
                 n_episodes,
                 n_step,
                 window_size,
                 transaction_cost=0):
        self.data_loader = data_loader
        self.dataset_name = dataset_name
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_memory_size = replay_memory_size
        self.feature_size = feature_size
        self.target_update = target_update
        self.n_episodes = n_episodes
        self.n_step = n_step
        self.transaction_cost = transaction_cost
        self.window_size = window_size

        self.STATE_MODE_OHLC = 1
        self.STATE_MODE_CANDLE_REP = 4  # trend + %body + %upper-shadow + %lower-shadow
        self.STATE_MODE_WINDOWED = 5  # window with k candles inside + the trend of those candles

        self.load_data()
        self.load_agents()

    def load_data(self):
        self.dataTrain_autoPatternExtractionAgent = DataAutoPatternExtractionAgent(self.data_loader.data_train,
                                                                                   self.STATE_MODE_OHLC,
                                                                                   'action_encoder_decoder', device,
                                                                                   self.gamma,
                                                                                   self.n_step, self.batch_size,
                                                                                   self.window_size,
                                                                                   self.transaction_cost)
        self.dataTest_autoPatternExtractionAgent = DataAutoPatternExtractionAgent(self.data_loader.data_test,
                                                                                  self.STATE_MODE_OHLC,
                                                                                  'action_encoder_decoder', device,
                                                                                  self.gamma,
                                                                                  self.n_step, self.batch_size,
                                                                                  self.window_size,
                                                                                  self.transaction_cost)
        self.dataTrain_patternBased = DataForPatternBasedAgent(self.data_loader.data_train, self.data_loader.patterns,
                                                               'action_deepRL',
                                                               device, self.gamma, self.n_step, self.batch_size,
                                                               self.transaction_cost)
        self.dataTest_patternBased = DataForPatternBasedAgent(self.data_loader.data_test, self.data_loader.patterns,
                                                              'action_deepRL',
                                                              device, self.gamma, self.n_step, self.batch_size,
                                                              self.transaction_cost)

        self.dataTrain_autoPatternExtractionAgent_candle_rep = DataAutoPatternExtractionAgent(
            self.data_loader.data_train,
            self.STATE_MODE_CANDLE_REP,
            'action_encoder_decoder',
            device,
            self.gamma, self.n_step, self.batch_size,
            self.window_size,
            self.transaction_cost)
        self.dataTest_autoPatternExtractionAgent_candle_rep = DataAutoPatternExtractionAgent(self.data_loader.data_test,
                                                                                             self.STATE_MODE_CANDLE_REP,
                                                                                             'action_encoder_decoder',
                                                                                             device,
                                                                                             self.gamma, self.n_step,
                                                                                             self.batch_size,
                                                                                             self.window_size,
                                                                                             self.transaction_cost)

        self.dataTrain_autoPatternExtractionAgent_windowed = DataAutoPatternExtractionAgent(self.data_loader.data_train,
                                                                                            self.STATE_MODE_WINDOWED,
                                                                                            'action_encoder_decoder',
                                                                                            device,
                                                                                            self.gamma, self.n_step,
                                                                                            self.batch_size,
                                                                                            self.window_size,
                                                                                            self.transaction_cost)
        self.dataTest_autoPatternExtractionAgent_windowed = DataAutoPatternExtractionAgent(self.data_loader.data_test,
                                                                                           self.STATE_MODE_WINDOWED,
                                                                                           'action_encoder_decoder',
                                                                                           device,
                                                                                           self.gamma, self.n_step,
                                                                                           self.batch_size,
                                                                                           self.window_size,
                                                                                           self.transaction_cost)

        self.dataTrain_sequential = DataSequential(self.data_loader.data_train,
                                                   'action_encoder_decoder', device, self.gamma,
                                                   self.n_step, self.batch_size, self.window_size,
                                                   self.transaction_cost)
        self.dataTest_sequential = DataSequential(self.data_loader.data_test,
                                                  'action_encoder_decoder', device, self.gamma,
                                                  self.n_step, self.batch_size, self.window_size, self.transaction_cost)

    def load_agents(self):
        self.dqn_pattern = DeepRL(self.data_loader,
                                  self.dataTrain_patternBased,
                                  self.dataTest_patternBased,
                                  self.dataset_name,
                                  state_mode,
                                  self.window_size,
                                  self.transaction_cost,
                                  BATCH_SIZE=self.batch_size,
                                  GAMMA=self.gamma,
                                  ReplayMemorySize=self.replay_memory_size,
                                  TARGET_UPDATE=self.target_update,
                                  n_step=self.n_step)

        self.dqn_vanilla = DeepRL(self.data_loader,
                                  self.dataTrain_autoPatternExtractionAgent,
                                  self.dataTest_autoPatternExtractionAgent,
                                  self.dataset_name,
                                  state_mode,
                                  self.window_size,
                                  self.transaction_cost,
                                  BATCH_SIZE=self.batch_size,
                                  GAMMA=self.gamma,
                                  ReplayMemorySize=self.replay_memory_size,
                                  TARGET_UPDATE=self.target_update,
                                  n_step=self.n_step)

        self.dqn_candle_rep = DeepRL(self.data_loader,
                                     self.dataTrain_autoPatternExtractionAgent_candle_rep,
                                     self.dataTest_autoPatternExtractionAgent_candle_rep,
                                     self.dataset_name,
                                     state_mode,
                                     self.window_size,
                                     self.transaction_cost,
                                     BATCH_SIZE=self.batch_size,
                                     GAMMA=self.gamma,
                                     ReplayMemorySize=self.replay_memory_size,
                                     TARGET_UPDATE=self.target_update,
                                     n_step=self.n_step)

        self.dqn_windowed = DeepRL(self.data_loader,
                                   self.dataTrain_autoPatternExtractionAgent_windowed,
                                   self.dataTest_autoPatternExtractionAgent_windowed,
                                   self.dataset_name,
                                   state_mode,
                                   self.window_size,
                                   self.transaction_cost,
                                   BATCH_SIZE=self.batch_size,
                                   GAMMA=self.gamma,
                                   ReplayMemorySize=self.replay_memory_size,
                                   TARGET_UPDATE=self.target_update,
                                   n_step=self.n_step)

        self.mlp_pattern = SimpleMLP(self.data_loader,
                                     self.dataTrain_patternBased,
                                     self.dataTest_patternBased,
                                     self.dataset_name,
                                     state_mode,
                                     self.window_size,
                                     self.transaction_cost,
                                     self.feature_size,
                                     BATCH_SIZE=self.batch_size,
                                     GAMMA=self.gamma,
                                     ReplayMemorySize=self.replay_memory_size,
                                     TARGET_UPDATE=self.target_update,
                                     n_step=self.n_step)

        self.mlp_vanilla = SimpleMLP(self.data_loader,
                                     self.dataTrain_autoPatternExtractionAgent,
                                     self.dataTest_autoPatternExtractionAgent,
                                     self.dataset_name,
                                     state_mode,
                                     self.window_size,
                                     self.transaction_cost,
                                     self.feature_size,
                                     BATCH_SIZE=self.batch_size,
                                     GAMMA=self.gamma,
                                     ReplayMemorySize=self.replay_memory_size,
                                     TARGET_UPDATE=self.target_update,
                                     n_step=self.n_step)

        self.mlp_candle_rep = SimpleMLP(self.data_loader,
                                        self.dataTrain_autoPatternExtractionAgent_candle_rep,
                                        self.dataTest_autoPatternExtractionAgent_candle_rep,
                                        self.dataset_name,
                                        state_mode,
                                        self.window_size,
                                        self.transaction_cost,
                                        self.feature_size,
                                        BATCH_SIZE=self.batch_size,
                                        GAMMA=self.gamma,
                                        ReplayMemorySize=self.replay_memory_size,
                                        TARGET_UPDATE=self.target_update,
                                        n_step=self.n_step)

        self.mlp_windowed = SimpleMLP(self.data_loader,
                                      self.dataTrain_autoPatternExtractionAgent_windowed,
                                      self.dataTest_autoPatternExtractionAgent_windowed,
                                      self.dataset_name,
                                      state_mode,
                                      self.window_size,
                                      self.transaction_cost,
                                      self.feature_size,
                                      BATCH_SIZE=self.batch_size,
                                      GAMMA=self.gamma,
                                      ReplayMemorySize=self.replay_memory_size,
                                      TARGET_UPDATE=self.target_update,
                                      n_step=self.n_step)

        self.cnn1d = SimpleCNN(self.data_loader,
                               self.dataTrain_autoPatternExtractionAgent,
                               self.dataTest_autoPatternExtractionAgent,
                               self.dataset_name,
                               state_mode,
                               self.window_size,
                               self.transaction_cost,
                               self.feature_size,
                               BATCH_SIZE=self.batch_size,
                               GAMMA=self.gamma,
                               ReplayMemorySize=self.replay_memory_size,
                               TARGET_UPDATE=self.target_update,
                               n_step=self.n_step)

        self.cnn2d = CNN2d(self.data_loader,
                           self.dataTrain_sequential,
                           self.dataTest_sequential,
                           self.dataset_name,
                           self.feature_size,
                           self.transaction_cost,
                           BATCH_SIZE=self.batch_size,
                           GAMMA=self.gamma,
                           ReplayMemorySize=self.replay_memory_size,
                           TARGET_UPDATE=self.target_update,
                           n_step=self.n_step,
                           window_size=self.window_size)

        self.gru = GRU(self.data_loader,
                       dataTrain_sequential,
                       dataTest_sequential,
                       self.dataset_name,
                       self.transaction_cost,
                       self.feature_size,
                       BATCH_SIZE=self.batch_size,
                       GAMMA=self.gamma,
                       ReplayMemorySize=self.replay_memory_size,
                       TARGET_UPDATE=self.target_update,
                       n_step=self.n_step,
                       window_size=self.window_size)

        self.deep_cnn = CNN(self.data_loader,
                            dataTrain_sequential,
                            dataTest_sequential,
                            self.dataset_name,
                            self.transaction_cost,
                            BATCH_SIZE=self.batch_size,
                            GAMMA=self.gamma,
                            ReplayMemorySize=self.replay_memory_size,
                            TARGET_UPDATE=self.target_update,
                            n_step=self.n_step,
                            window_size=self.window_size)

        self.cnn_gru = CNN_GRU(self.data_loader,
                               dataTrain_sequential,
                               dataTest_sequential,
                               self.dataset_name,
                               self.transaction_cost,
                               self.feature_size,
                               BATCH_SIZE=self.batch_size,
                               GAMMA=self.gamma,
                               ReplayMemorySize=self.replay_memory_size,
                               TARGET_UPDATE=self.target_update,
                               n_step=self.n_step,
                               window_size=self.window_size)

        self.cnn_attn = CNN_ATTN(self.data_loader,
                                 dataTrain_sequential,
                                 dataTest_sequential,
                                 self.dataset_name,
                                 self.transaction_cost,
                                 self.feature_size,
                                 BATCH_SIZE=self.batch_size,
                                 GAMMA=self.gamma,
                                 ReplayMemorySize=self.replay_memory_size,
                                 TARGET_UPDATE=self.target_update,
                                 n_step=self.n_step,
                                 window_size=self.window_size)

    def train(self):
        self.dqn_pattern.train(self.n_episodes)
        self.dqn_vanilla.train(self.n_episodes)
        self.dqn_candle_rep.train(self.n_episodes)
        self.dqn_windowed.train(self.n_episodes)
        self.mlp_pattern.train(self.n_episodes)
        self.mlp_vanilla.train(self.n_episodes)
        self.mlp_candle_rep.train(self.n_episodes)
        self.mlp_windowed.train(self.n_episodes)
        self.cnn1d.train(self.n_episodes)
        self.cnn2d.train(self.n_episodes)
        self.gru.train(self.n_episodes)
        self.deep_cnn.train(self.n_episodes)
        self.cnn_gru.train(self.n_episodes)
        self.cnn_attn.train(self.n_episodes)


if __name__ == '__main__':
    self.gamma = [0.9, 0.8, 0.7]
    self.batch_size = [16, 64, 256]
    self.replay_memory_size = [16, 64, 256]
    self.n_step = [4, 8, 16]
    self.window_size = 3  # Default value for the experiments of the first paper.

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
