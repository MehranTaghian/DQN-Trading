import torch
import torch.optim as optim

from EncoderDecoderAgent.CNN2D.Seq2SeqModel import Seq2Seq
from EncoderDecoderAgent.CNN2D.Encoder import Encoder
from EncoderDecoderAgent.CNN2D.Decoder import Decoder

from EncoderDecoderAgent.BaseTrain import BaseTrain

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Train(BaseTrain):
    def __init__(self,
                 data_loader,
                 data_train,
                 data_test,
                 dataset_name,
                 decoder_input_size,
                 transaction_cost,
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
        @param decoder_input_size: size of the feature vector output of the encoder
        """
        super(Train, self).__init__(data_loader,
                                    data_train,
                                    data_test,
                                    dataset_name,
                                    'CNN2D',
                                    transaction_cost,
                                    BATCH_SIZE,
                                    GAMMA,
                                    ReplayMemorySize,
                                    TARGET_UPDATE,
                                    n_step,
                                    window_size)

        self.decoder_input_size = decoder_input_size

        self.encoder = Encoder(self.decoder_input_size, data_train.state_size, window_size)
        self.policy_decoder = Decoder(self.decoder_input_size)
        self.target_decoder = Decoder(self.decoder_input_size)

        self.policy_net = Seq2Seq(self.encoder, self.policy_decoder).to(device)
        self.target_net = Seq2Seq(self.encoder, self.target_decoder).to(device)

        self.target_decoder.load_state_dict(self.policy_decoder.state_dict())
        self.target_decoder.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())

        test_encoder = Encoder(self.decoder_input_size, data_train.state_size, window_size).to(device)
        test_decoder = Decoder(self.decoder_input_size).to(device)

        self.test_net = Seq2Seq(test_encoder, test_decoder)
