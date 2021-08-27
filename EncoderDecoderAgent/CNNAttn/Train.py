import torch
import torch.optim as optim

from EncoderDecoderAgent.CNNAttn.Seq2SeqModel import Seq2Seq
from EncoderDecoderAgent.CNNAttn.Encoder import Encoder
from EncoderDecoderAgent.CNNAttn.Decoder import Decoder
from EncoderDecoderAgent.CNNAttn.Attention import AttentionLayer
from EncoderDecoderAgent.BaseTrain import BaseTrain


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Train(BaseTrain):
    def __init__(self, data_loader, data_train, data_test, dataset_name, transaction_cost, attn_output_size=64,
                 BATCH_SIZE=30,
                 GAMMA=0.7, EPS=0.1, ReplayMemorySize=50, TARGET_UPDATE=5, n_actions=3, n_step=10, window_size=20):
        """
        :param TARGET_UPDATE: Every TARGET_UPDATE iterations, we give the weights of the Policy network to the Target
                                network.
        :param n_step: n in n-step SARSA
        """
        super(Train, self).__init__(data_loader, data_train, data_test, dataset_name, 'CNN-ATTN', transaction_cost,
                                    BATCH_SIZE, GAMMA, EPS, ReplayMemorySize, TARGET_UPDATE,
                                    n_actions, n_step, window_size)

        self.encoder = Encoder(self.data_train.state_size)
        self.attention = AttentionLayer(self.window_size, attn_output_size)
        self.policy_decoder = Decoder(attn_output_size)
        self.target_decoder = Decoder(attn_output_size)

        self.policy_net = Seq2Seq(self.encoder, self.attention, self.policy_decoder).to(device)
        self.target_net = Seq2Seq(self.encoder, self.attention, self.target_decoder).to(device)

        self.target_decoder.load_state_dict(self.policy_decoder.state_dict())
        self.target_decoder.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer_encoder, step_size=5, gamma=0.1)

        test_encoder = Encoder(self.data_train.state_size).to(device)
        test_attn = AttentionLayer(self.window_size, attn_output_size).to(device)
        test_decoder = Decoder(attn_output_size).to(device)

        self.test_net = Seq2Seq(test_encoder, test_attn, test_decoder)

