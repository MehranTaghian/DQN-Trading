import torch
import torch.optim as optim

from DeepRLAgent.SimpleCNNEncoder.Seq2SeqModel import Seq2Seq
from DeepRLAgent.SimpleCNNEncoder.Decoder import Decoder
from DeepRLAgent.SimpleCNNEncoder.Encoder import Encoder
from DeepRLAgent.BaseTrain import BaseTrain

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Train(BaseTrain):
    def __init__(self, data_loader, data_train, data_test, dataset_name, state_mode=1, window_size=1,
                 transaction_cost=0.0, n_classes=64, BATCH_SIZE=30, GAMMA=0.7, EPS=0.1,
                 ReplayMemorySize=50, TARGET_UPDATE=5, n_actions=3, n_step=10):
        """
        :param TARGET_UPDATE: Every TARGET_UPDATE iterations, we give the weights of the Policy network to the Target
                                network.
        :param n_step: n in n-step SARSA
        """
        super(Train, self).__init__(data_loader, data_train, data_test, dataset_name, 'SimpleCNNEncoder', state_mode,
                                    window_size, transaction_cost, BATCH_SIZE, GAMMA,
                                    EPS, ReplayMemorySize, TARGET_UPDATE, n_actions, n_step)

        # self.state_mode = state_mode
        # self.window_size = window_size

        self.encoder = Encoder(n_classes, data_train.state_size).to(device)
        self.policy_decoder = Decoder(n_classes, n_actions).to(device)
        self.target_decoder = Decoder(n_classes, n_actions).to(device)

        self.policy_net = Seq2Seq(self.encoder, self.policy_decoder).to(device)
        self.target_net = Seq2Seq(self.encoder, self.target_decoder).to(device)

        self.target_decoder.load_state_dict(self.policy_decoder.state_dict())
        self.target_decoder.eval()

        # optimizer = optim.RMSprop(policy_net.parameters())
        self.optimizer = optim.Adam(self.policy_net.parameters())
        # self.optimizer_encoder = optim.Adam(self.encoder.parameters())
        # self.optimizer_policy_net = optim.Adam(self.policy_net.parameters())

        test_encoder = Encoder(n_classes, self.data_train.state_size).to(device)
        test_decoder = Decoder(n_classes, self.n_actions).to(device)

        self.test_net = Seq2Seq(test_encoder, test_decoder)
        self.test_net.to(device)
