import torch
import torch.optim as optim
from EncoderDecoderAgent.GRU.Seq2SeqModel import Seq2Seq
from EncoderDecoderAgent.GRU.Decoder import Decoder
from EncoderDecoderAgent.GRU.Encoder import EncoderRNN
from EncoderDecoderAgent.GRU.Attention import AttentionLayer

from EncoderDecoderAgent.BaseTrain import BaseTrain
from torch.nn import init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Train(BaseTrain):
    def __init__(self, data_loader, data_train, data_test, dataset_name, transaction_cost, hidden_size=50,
                 BATCH_SIZE=30,
                 GAMMA=0.7, EPS=0.1, ReplayMemorySize=50, TARGET_UPDATE=5, n_actions=3, n_step=10, window_size=20):
        """
        :param TARGET_UPDATE: Every TARGET_UPDATE iterations, we give the weights of the Policy network to the Target
                                network.
        :param n_step: n in n-step SARSA
        """
        super(Train, self).__init__(data_loader, data_train, data_test, dataset_name, 'GRU', transaction_cost,
                                    BATCH_SIZE,
                                    GAMMA, EPS, ReplayMemorySize, TARGET_UPDATE, n_actions, n_step, window_size)
        self.hidden_size = hidden_size

        self.encoder = EncoderRNN(self.data_train.state_size, self.hidden_size, device).to(device)
        self.attention = AttentionLayer(self.hidden_size, self.window_size, device).to(device)
        self.policy_decoder = Decoder(self.hidden_size, self.n_actions).to(device)
        self.target_decoder = Decoder(self.hidden_size, self.n_actions).to(device)

        # for _, param in self.encoder.named_parameters():
        #     init.normal(param, 0, 1)

        self.policy_net = Seq2Seq(self.encoder, self.attention, self.policy_decoder).to(device)
        self.target_net = Seq2Seq(self.encoder, self.attention, self.target_decoder).to(device)

        self.optimizer = optim.Adam(self.policy_net.parameters())
        # self.optimizer_encoder = optim.Adam(self.encoder.parameters(), lr=1e3)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer_encoder, step_size=5, gamma=0.1)
        # self.optimizer_policy_net = optim.Adam(self.policy_net.parameters())

        self.target_decoder.load_state_dict(self.policy_decoder.state_dict())
        self.target_decoder.eval()

        test_encoder = EncoderRNN(self.data_train.state_size, self.hidden_size, device).to(device)
        test_attention = AttentionLayer(self.hidden_size, self.window_size, device).to(device)
        test_decoder = Decoder(self.hidden_size, self.n_actions).to(device)

        self.test_net = Seq2Seq(test_encoder, test_attention, test_decoder)
