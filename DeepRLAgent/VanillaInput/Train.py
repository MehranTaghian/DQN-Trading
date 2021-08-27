import torch
import torch.optim as optim
from DeepRLAgent.VanillaInput.DeepQNetwork import DQN
from DeepRLAgent.BaseTrain import BaseTrain

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Train(BaseTrain):
    def __init__(self, data_loader, data_train, data_test, dataset_name, state_mode=1, window_size=1,
                 transaction_cost=0.0, BATCH_SIZE=30, GAMMA=0.7, EPS=0.1, ReplayMemorySize=50,
                 TARGET_UPDATE=5, n_actions=3, n_step=10):
        """
        :param TARGET_UPDATE: Every TARGET_UPDATE iterations, we give the weights of the Policy network to the Target
                                network.
        :param n_step: n in n-step SARSA
        """
        super(Train, self).__init__(data_loader, data_train, data_test, dataset_name, 'DeepRL', state_mode, window_size,
                                    transaction_cost, BATCH_SIZE, GAMMA, EPS, ReplayMemorySize,
                                    TARGET_UPDATE, n_actions, n_step)

        self.policy_net = DQN(data_train.state_size, n_actions).to(device)
        self.target_net = DQN(data_train.state_size, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # optimizer = optim.RMSprop(policy_net.parameters())
        self.optimizer = optim.Adam(self.policy_net.parameters())

        self.test_net = DQN(self.data_train.state_size, self.n_actions)
        self.test_net.to(device)
