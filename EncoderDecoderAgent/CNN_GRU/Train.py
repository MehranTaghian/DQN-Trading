import torch
import torch.optim as optim

from EncoderDecoderAgent.CNN_GRU.Seq2SeqModel import Seq2Seq
from EncoderDecoderAgent.CNN_GRU.Encoder import Encoder
from EncoderDecoderAgent.CNN_GRU.Decoder import Decoder
from EncoderDecoderAgent.BaseTrain import BaseTrain

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Train(BaseTrain):
    def __init__(self,
                 data_loader,
                 data_train,
                 data_test,
                 dataset_name,
                 transaction_cost,
                 hidden_size=64,
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
        @param hidden_size: size of the output feature vector from the encoder
        @param data_loader: The data loader here is to only access the start_data, end_data and split point in order to
            name the result file of the experiment
        @param data_train: of type DataAutoPatternExtractionAgent
        @param data_test: of type DataAutoPatternExtractionAgent
        @param dataset_name: for using in the name of the result file
        @param window_size: for using in the name of the result file
        @param transaction_cost: for using in the name of the result file
        @param BATCH_SIZE: batch size for batch training
        @param GAMMA: in the algorithm
        @param ReplayMemorySize: size of the replay buffer
        @param TARGET_UPDATE: hard update policy network into target network every TARGET_UPDATE iterations
        @param n_step: for using in the name of the result file
        """

        super(Train, self).__init__(data_loader,
                                    data_train,
                                    data_test,
                                    dataset_name,
                                    'CNN-GRU',
                                    transaction_cost,
                                    BATCH_SIZE,
                                    GAMMA,
                                    ReplayMemorySize,
                                    TARGET_UPDATE,
                                    n_step,
                                    window_size)

        self.encoder = Encoder(self.window_size, hidden_size, device)
        self.policy_decoder = Decoder(hidden_size)
        self.target_decoder = Decoder(hidden_size)

        self.policy_net = Seq2Seq(self.encoder, self.policy_decoder).to(device)
        self.target_net = Seq2Seq(self.encoder, self.target_decoder).to(device)

        self.target_decoder.load_state_dict(self.policy_decoder.state_dict())
        self.target_decoder.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())

        test_encoder = Encoder(self.window_size, hidden_size, device).to(device)
        test_decoder = Decoder(hidden_size).to(device)

        self.test_net = Seq2Seq(test_encoder, test_decoder)
