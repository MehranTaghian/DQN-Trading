from DataLoader.DataLoader import YahooFinanceDataLoader
from DataLoader.DataAutoPatternExtractionAgent import DataAutoPatternExtractionAgent
from DeepRLAgent.SimpleCNNEncoder.Train import Train
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 10
GAMMA = 0.7
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 200
EPS = 0.1
ReplayMemorySize = 20
window_size = 20

TARGET_UPDATE = 5
n_actions = 3
n_step = 10  # in n-step SARSA
n_classes = 128

# State Mode
state_mode = 1  # OHLC
# state_mode = 2  # OHLC + trend
# state_mode = 3  # OHLC + trend + %body + %upper-shadow + %lower-shadow
# state_mode = 4  # trend + %body + %upper-shadow + %lower-shadow
# state_mode = 5  # window with k candles inside + the trend of those candles


# BTC-USD

# DATASET_NAME = 'BTC-USD'
# DATASET_FOLDER = r'BTC-USD'
# FILE = r'BTC-USD.csv'
# data_loader = YahooFinanceDataLoader(DATASET_FOLDER, FILE, '2018-01-01', load_from_file=True)
# transaction_cost = 0

# AAPL

# DATASET_NAME = 'AAPL'
# DATASET_FOLDER = r'AAPL'
# FILE = r'AAPL.csv'
# data_loader = YahooFinanceDataLoader(DATASET_FOLDER, FILE, '2018-01-01', begin_date='2010-01-01', end_date='2020-08-24'
#                                      , load_from_file=True)
# transaction_cost = 0.0


# GOOGL

# DATASET_NAME = 'GOOGL'
# DATASET_FOLDER = 'GOOGL'
# FILE = 'GOOGL.csv'
# data_loader = YahooFinanceDataLoader(DATASET_FOLDER, FILE, '2018-01-01', load_from_file=True)
# transaction_cost = 0.0

# KSS

DATASET_NAME = 'KSS'
DATASET_FOLDER = 'KSS'
FILE = 'KSS.csv'
data_loader = YahooFinanceDataLoader(DATASET_FOLDER, FILE, split_point='2018-01-01', load_from_file=True)
transaction_cost = 0.0

# AAPL

# DATASET_NAME = 'AAPL'
# DATASET_FOLDER = r'AAPL'
# FILE = r'AAPL.csv'
# data_loader = YahooFinanceDataLoader(DATASET_FOLDER, FILE, '2016-01-01', begin_date='2008-01-01', end_date='2018-12-20'
#                                      , load_from_file=True)
# transaction_cost = 0.0

# AAL

# DATASET_NAME = 'AAL'
# DATASET_FOLDER = r'AAL'
# FILE = r'AAL.csv'
# data_loader = YahooFinanceDataLoader(DATASET_FOLDER, FILE, '2018-01-01', load_from_file=True)

# # DJI
#
# DATASET_NAME = 'DJI'
# DATASET_FOLDER = r'DJI'
# FILE = r'DJI.csv'
# data_loader = YahooFinanceDataLoader(DATASET_FOLDER, FILE, '2016-01-01', begin_date='2009-01-01', end_date='2018-09-30',
#                                      load_from_file=True)

# S&P

# DATASET_NAME = 'S&P'
# DATASET_FOLDER = 'S&P'
# FILE = 'S&P.csv'
# data_loader = YahooFinanceDataLoader(DATASET_FOLDER, FILE, split_point=2000, end_date='2018-09-25', load_from_file=True)
# transaction_cost = 0.001

# AMD

# DATASET_NAME = 'AMD'
# DATASET_FOLDER = 'AMD'
# FILE = 'AMD.csv'
# data_loader = YahooFinanceDataLoader(DATASET_FOLDER, FILE, split_point=2000, end_date='2018-09-25', load_from_file=True)

# HSI

# DATASET_NAME = 'HSI'
# DATASET_FOLDER = 'HSI'
# FILE = 'HSI.csv'
# data_loader = YahooFinanceDataLoader(DATASET_FOLDER, FILE, split_point='2005-01-01', load_from_file=True)
# transaction_cost = 0.0

# S&P

# DATASET_NAME = 'S&P'
# DATASET_FOLDER = 'S&P'
# FILE = 'S&P.csv'
# data_loader = YahooFinanceDataLoader(DATASET_FOLDER, FILE, split_point='2005-01-01', begin_date='2001-01-01',
#                                      end_date='2015-12-31', load_from_file=True)
# transaction_cost = 0

dataTrain = DataAutoPatternExtractionAgent(data_loader.data_train, state_mode,
                                           'action_encoder_decoder', device, GAMMA,
                                           n_step, BATCH_SIZE, window_size, transaction_cost=transaction_cost)
dataTest = DataAutoPatternExtractionAgent(data_loader.data_test, state_mode,
                                          'action_encoder_decoder', device, GAMMA,
                                          n_step, BATCH_SIZE, window_size, transaction_cost=transaction_cost)

deepRLAgent = Train(data_loader, dataTrain, dataTest, DATASET_NAME, state_mode, window_size, transaction_cost,
                    n_classes, BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA, EPS=EPS,
                    ReplayMemorySize=ReplayMemorySize,
                    TARGET_UPDATE=TARGET_UPDATE,
                    n_actions=n_actions,
                    n_step=n_step)

# tb = SummaryWriter()
# deepRLAgent.train(30, tb)
# tb.close()

deepRLAgent.train(10)
file_name = None

initial_investment = 1000

deepRLAgent.test(file_name=file_name, action_name=dataTrain.action_name, initial_investment=initial_investment,
                 test_type='train')
deepRLAgent.test(file_name=file_name, action_name=dataTrain.action_name, initial_investment=initial_investment,
                 test_type='test')
