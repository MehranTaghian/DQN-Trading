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
    def __init__(self, data_loader,
                 data_train,
                 data_test,
                 dataset_name,
                 transaction_cost,
                 hidden_size=50,
                 BATCH_SIZE=30,
                 GAMMA=0.7,
                 ReplayMemorySize=50,
                 TARGET_UPDATE=5,
                 n_step=10,
                 window_size=20):
        """
        This class is inherited from the BaseTrain class to initialize networks and other stuff that are specific to this
        model. For those parameters in the following explanation that I wrote: "for using in the name of the result file"
        the effect of those parameters has been applied in the Data class and are mentioned here only for begin used as
        part of the experiment's result filename.
        TODO: For this specific model, I also tried to use attention mechanism but it didn't work as expected.
         Therefore I commented out the attention mechanism in the Seq2SeqModel.py.
        @param data_loader: The data loader here is to only access the start_data, end_data and split point in order to
            name the result file of the experiment
        @param data_train: of type DataAutoPatternExtractionAgent
        @param data_test: of type DataAutoPatternExtractionAgent
        @param dataset_name: for using in the name of the result file
        @param window_size: for using in the name of the result file
        @param transaction_cost: for using in the name of the result file
        @param BATCH_SIZE: batch size for batch training
        @param GAMMA: in the algorithm
        @param hidden_size: size of the output feature vector from the encoder
        @param ReplayMemorySize: size of the replay buffer
        @param TARGET_UPDATE: hard update policy network into target network every TARGET_UPDATE iterations
        @param n_step: for using in the name of the result file
        """
        super(Train, self).__init__(data_loader,
                                    data_train,
                                    data_test,
                                    dataset_name,
                                    'GRU',
                                    transaction_cost,
                                    BATCH_SIZE,
                                    GAMMA,
                                    ReplayMemorySize,
                                    TARGET_UPDATE,
                                    n_step,
                                    window_size)
        self.hidden_size = hidden_size

        self.encoder = EncoderRNN(self.data_train.state_size, self.hidden_size, device).to(device)
        self.attention = AttentionLayer(self.hidden_size, self.window_size, device).to(device)
        self.policy_decoder = Decoder(self.hidden_size).to(device)
        self.target_decoder = Decoder(self.hidden_size).to(device)

        self.policy_net = Seq2Seq(self.encoder, self.attention, self.policy_decoder).to(device)
        self.target_net = Seq2Seq(self.encoder, self.attention, self.target_decoder).to(device)

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.target_decoder.load_state_dict(self.policy_decoder.state_dict())
        self.target_decoder.eval()

        test_encoder = EncoderRNN(self.data_train.state_size, self.hidden_size, device).to(device)
        test_attention = AttentionLayer(self.hidden_size, self.window_size, device).to(device)
        test_decoder = Decoder(self.hidden_size).to(device)

        self.test_net = Seq2Seq(test_encoder, test_attention, test_decoder)
