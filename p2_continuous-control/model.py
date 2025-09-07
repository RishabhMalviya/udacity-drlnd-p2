import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, s_size=33, a_size=4, h_size=None):
        super(Actor, self).__init__()

        if not h_size:
            h_size = 256

        self.fc1 = nn.Linear(s_size, h_size)
        self.fc_means = nn.Linear(h_size, a_size)
        self.fc_log_variances = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        means = F.tanh(self.fc_means(x))
        log_variances = F.relu(self.fc_log_variances(x))

        stds = (0.5 * log_variances).exp()
        dist = Normal(means, stds)
        actions = dist.rsample()

        return actions, dist.log_prob(actions).sum(dim=-1)


class Critic(nn.Module):
    def __init__(self, s_size=33, a_size=4, h_size=None):
        super(Critic, self).__init__()

        if not h_size:
            h_size = 256

        self.fcs1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size + a_size, h_size // 2)
        self.fc3 = nn.Linear(h_size // 2, 1)

    def forward(self, state, action):
        x_s = F.relu(self.fcs1(state))
        x = torch.cat((x_s, action), dim=-1)
        
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x