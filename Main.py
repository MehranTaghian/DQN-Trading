# Importing DataLoaders for each model. These models include rule-based, vanilla DQN and encoder-decoder DQN.
from DataLoader.DataLoader import YahooFinanceDataLoader
from DataLoader.DataForPatternBasedAgent import DataForPatternBasedAgent
from DataLoader.DataAutoPatternExtractionAgent import DataAutoPatternExtractionAgent
from DataLoader.DataSequential import DataSequential

from DeepRLAgent.MLPEncoder.Train import Train as SimpleMLP
from DeepRLAgent.SimpleCNNEncoder.Train import Train as SimpleCNN
from EncoderDecoderAgent.GRU.Train import Train as gru
from EncoderDecoderAgent.CNN.Train import Train as cnn
from EncoderDecoderAgent.CNN2D.Train import Train as cnn2d
from EncoderDecoderAgent.CNNAttn.Train import Train as cnn_attn
from EncoderDecoderAgent.CNN_GRU.Train import Train as cnn_gru

# Imports for Deep RL Agent
from DeepRLAgent.VanillaInput.Train import Train as DeepRL
# Imports for RL Agent with n-step SARSA
from RLAgent.Train import Train as RLTrain

# Imports for Rule-Based
from PatternDetectionInCandleStick.LabelPatterns import label_candles
from PatternDetectionInCandleStick.Evaluation import Evaluation

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


def sensitivity_run(data_loader, gamma, batch_size, replay_memory_size, n_step, window_size, transaction_cost=0):
    state_mode = 1
    dataTrain_autoPatternExtractionAgent = DataAutoPatternExtractionAgent(data_loader.data_train, state_mode,
                                                                          'action_encoder_decoder', device, gamma,
                                                                          n_step, batch_size, window_size,
                                                                          transaction_cost)
    dataTest_autoPatternExtractionAgent = DataAutoPatternExtractionAgent(data_loader.data_test, state_mode,
                                                                         'action_encoder_decoder', device, gamma,
                                                                         n_step, batch_size, window_size,
                                                                         transaction_cost)
    dataTrain_patternBased = DataForPatternBasedAgent(data_loader.data_train, data_loader.patterns, 'action_deepRL',
                                                      device, gamma, n_step, batch_size, transaction_cost)
    dataTest_patternBased = DataForPatternBasedAgent(data_loader.data_test, data_loader.patterns, 'action_deepRL',
                                                     device, gamma, n_step, batch_size, transaction_cost)


if __name__ == '__main__':
    gamma = [0.9, 0.8, 0.7]
    batch_size = [16, 64, 256]
    replay_memory_size = [16, 64, 256]
    n_step = [4, 8, 16]
    window_size = 3  # Default value for the experiments of the first paper.

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
