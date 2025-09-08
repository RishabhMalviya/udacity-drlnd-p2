import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic


GAMMA = 0.9
LR_ACTOR = 1e-4
LR_CRITIC = 1e-4


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size=33, action_size=4, seed=42):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        # Hyperparameters
        self.GAMMA = GAMMA
        self.LR_ACTOR = LR_ACTOR
        # self.LR_CRITIC = LR_CRITIC
        self.BASELINE_REWARD = 0.05

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Actor Network
        self.actor_nn = Actor(self.state_size, self.action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_nn.parameters(), lr=self.LR_ACTOR)

        # # Critic Network
        # self.critic_nn = Critic(self.state_size, self.action_size).to(device)
        # self.critic_optimizer = optim.Adam(self.critic_nn.parameters(), lr=self.LR_CRITIC)

    def reset(self):
        self.rewards = []
        self.log_probs = []

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        actions, log_prob = self.actor_nn.forward(state)
        
        self.log_probs.append(log_prob)
        
        return actions.detach().cpu().numpy()[0]

    def learn(self):
        """
        Update policy and value netowrk parameters at the end of the episode.
        """
        # Compute the discounted returns
        self.rewards = np.array(self.rewards)
        discounts = np.array([self.GAMMA**i for i in range(len(self.rewards))] + [0.0]) # Extra 0.0 for indexing convenience
        returns = np.array([
            sum(self.rewards[i-1:]*discounts[:-i]) - sum(discounts[:-i])*self.BASELINE_REWARD
            for i in range(1, len(self.rewards)+1)
        ])

        # Compute the Policy Gradient
        actor_policy_loss = torch.cat([ -log_probs * returns for log_probs, returns in zip(self.log_probs, returns) ]).sum()        
        actor_policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()

        # Reset episode variables
        self.reset()

    def save(self, filename='checkpoint.pth'):
        torch.save(self.actor_nn.state_dict(), filename)

    def load_weights(self, filename='checkpoint.pth'):
        self.actor_nn.load_state_dict(torch.load(filename))
        self.actor_nn.eval()
