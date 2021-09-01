from DataLoader.DataLoader import YahooFinanceDataLoader
from DataLoader.DataSequential import DataSequential
from EncoderDecoderAgent.CNN2D.Train import Train
from torch.utils.tensorboard import SummaryWriter
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 10
GAMMA = 0.7
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 200
EPS = 0.1
ReplayMemorySize = 20
window_size = 3
decoder_input_size = 128

TARGET_UPDATE = 5
n_actions = 3
n_step = 10  # in n-step SARSA

# BTC-USD

# DATASET_NAME = 'BTC-USD'
# DATASET_FOLDER = r'BTC-USD'
# FILE = r'BTC-USD.csv'
# data_loader = YahooFinanceDataLoader(DATASET_FOLDER, FILE, '2018-01-01', load_from_file=True)
# transaction_cost = 0.0

# AAPL

# DATASET_NAME = 'AAPL'
# DATASET_FOLDER = r'AAPL'
# FILE = r'AAPL.csv'
# data_loader = YahooFinanceDataLoader(DATASET_FOLDER, FILE, '2018-01-01', begin_date='2010-01-01', end_date='2020-08-24'
#                                      , load_from_file=True)
# transaction_cost = 0.0

# AAPL

# DATASET_NAME = 'AAPL'
# DATASET_FOLDER = r'AAPL'
# FILE = r'AAPL.csv'
# data_loader = YahooFinanceDataLoader(DATASET_FOLDER, FILE, '2016-01-01', begin_date='2008-01-01', end_date='2018-12-20'
#                                      , load_from_file=True)
# transaction_cost = 0.0

# GE

# DATASET_NAME = 'GE'
# DATASET_FOLDER = r'GE'
# FILE = r'GE.csv'
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
# transaction_cost = 0.0

# S&P

# DATASET_NAME = 'S&P'
# DATASET_FOLDER = 'S&P'
# FILE = 'S&P.csv'
# data_loader = YahooFinanceDataLoader(DATASET_FOLDER, FILE, split_point=2000, end_date='2018-09-25', load_from_file=True)
# transaction_cost = 0.001

# GOOGL

# DATASET_NAME = 'GOOGL'
# DATASET_FOLDER = 'GOOGL'
# FILE = 'GOOGL.csv'
# data_loader = YahooFinanceDataLoader(DATASET_FOLDER, FILE, '2018-01-01', load_from_file=True)
#
# transaction_cost = 0.0

# KSS

DATASET_NAME = 'KSS'
DATASET_FOLDER = 'KSS'
FILE = 'KSS.csv'
data_loader = YahooFinanceDataLoader(DATASET_FOLDER, FILE, split_point='2018-01-01', load_from_file=True)
transaction_cost = 0.0

# AMD

# DATASET_NAME = 'AMD'
# DATASET_FOLDER = 'AMD'
# FILE = 'AMD.csv'
# data_loader = YahooFinanceDataLoader(DATASET_FOLDER, FILE, split_point=2000, end_date='2018-09-25', load_from_file=True)
# transaction_cost = 0.0


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

dataTrain = DataSequential(data_loader.data_train,
                               'action_encoder_decoder', device, GAMMA,
                           n_step, BATCH_SIZE, window_size, transaction_cost=transaction_cost)
dataTest = DataSequential(data_loader.data_test,
                              'action_encoder_decoder', device, GAMMA,
                          n_step, BATCH_SIZE, window_size, transaction_cost=transaction_cost)

deepRLAgent = Train(data_loader, dataTrain, dataTest, DATASET_NAME, decoder_input_size, transaction_cost,
                    BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA, EPS=EPS,
                    ReplayMemorySize=ReplayMemorySize,
                    TARGET_UPDATE=TARGET_UPDATE,
                    n_actions=n_actions,
                    n_step=n_step,
                    window_size=window_size)

# tb = SummaryWriter()
# deepRLAgent.train(30, tb)
# tb.close()
# file_name = None

deepRLAgent.train(10)
file_name = None

initial_investment = 1000

# file_name = 'KSS; DATA_KIND(LSTMSequential); Dates(None, 2018-01-01, None); CNN2D; TC(0.0); WindowSize(3); BATCH_SIZE10; GAMMA0.7; EPS0.1; REPLAY_MEMORY_SIZE20; C5; N_SARSA10(1).pkl'

deepRLAgent.test(file_name=file_name, action_name=dataTrain.action_name, initial_investment=initial_investment,
                 test_type='train')
deepRLAgent.test(file_name=file_name, action_name=dataTrain.action_name, initial_investment=initial_investment,
                 test_type='test')
