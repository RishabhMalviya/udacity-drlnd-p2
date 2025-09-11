import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, s_size=33, a_size=4):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(s_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_means = nn.Linear(128, a_size)
        self.fc_log_variances = nn.Linear(128, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        means = self.fc_means(x)

        log_variances = self.fc_log_variances(x)
        stds = (0.5 * log_variances).exp()

        dist = Normal(means, stds)
        actions = dist.rsample()

        return actions, dist.log_prob(actions).sum(dim=-1)


class Critic(nn.Module):
    def __init__(self, s_size=33):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(s_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))

        return x