import numpy as np

import torch
from torch import nn, optim

from utils import ReplayBuffer, device
from model import Actor, Critic



class DummyAgent:
    def __init__(self, state_size=33, action_size=4, num_agents=20):
        self.STATE_SIZE = state_size
        self.ACTION_SIZE = action_size
        self.NUM_AGENTS = num_agents

    def act(self, state):
        actions = np.random.randn(self.NUM_AGENTS, self.ACTION_SIZE)
        actions = np.clip(actions, -1, 1)
        
        return actions
    
    def step(self, state, action, reward, next_state, done):
        pass

    def checkpoint(self):
        pass
    
    def load_networks(self):
        pass

    def save_networks(self):
        pass


class Agent:
    def __init__(self, num_agents=20, state_size=33, action_size=4):
        # Hyperparameters
        self.NUM_AGENTS = num_agents
        self.STATE_SIZE = state_size
        self.ACTION_SIZE = action_size

        self.NOISE_SCALE = 0.2
        self.TAU = 1e-3
        self.GAMMA = 0.9
        self.ACTOR_LR = 1e-4
        self.CRITIC_LR = 1e-4

        self.LEARN_EVERY = 1
        self.BATCH_SIZE = 2048

        # Actor
        self.local_actor = Actor(state_size, action_size).to(device)

        self.target_actor = Actor(state_size, action_size).to(device)
        self.target_actor.load_state_dict(self.local_actor.state_dict())
        self.target_actor.eval()

        self.actor_optimizer = optim.Adam(self.local_actor.parameters(), lr=self.ACTOR_LR)

        # Critic
        self.local_critic = Critic(state_size, action_size).to(device)

        self.target_critic = Critic(state_size, action_size).to(device)
        self.target_critic.load_state_dict(self.local_critic.state_dict())
        self.target_critic.eval()

        self.critic_loss_fn = nn.MSELoss()

        self.critic_optimizer = optim.Adam(self.local_critic.parameters(), lr=self.CRITIC_LR)

        # Replay Memory
        self.replay_buffer = ReplayBuffer(batch_size=self.BATCH_SIZE)

        # State Variables
        self.t_step = 0

    def act(self, state):
        state = torch.from_numpy(state).float().to(device)

        self.local_actor.eval()
        with torch.no_grad():
            actions = self.local_actor(state).cpu().data.numpy()
        self.local_actor.train()

        noise = np.random.normal(loc=0.0, scale=self.NOISE_SCALE, size=actions.shape)
        actions += noise
        actions = np.clip(actions, -1, 1)
        
        return actions

    def step(self, state, action, reward, next_state, done):
        self.t_step += 1

        for i in range(self.NUM_AGENTS):
            self.replay_buffer.add(state[i], action[i], reward[i], next_state[i], done[i])

        if (len(self.replay_buffer) > self.BATCH_SIZE) and (self.t_step % self.LEARN_EVERY == 0):
            self._learn()
            self.t_step = 0

    def _learn(self):
        def _soft_update(local_model, target_model):
            for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)

        # ------ Sample Experiences ------ #
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        # ------ Train Local Critic ------ #
        # Calculate Q-Targets
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_states_value_estimates = self.target_critic(next_states, next_actions)
            Q_targets = rewards + (self.GAMMA * (1 - dones) * next_states_value_estimates)
        # Compute Q-Estimates
        Q_estimates = self.local_critic(states, actions)
        # Compute Critic Loss (MSE between Q-Targets and Q-Estimates)
        msbe_loss = self.critic_loss_fn(Q_estimates, Q_targets)
        # Backpropagate and Optimize
        self.critic_optimizer.zero_grad()
        msbe_loss.backward()
        self.critic_optimizer.step()

        # ------ Train Local Actor ------ #
        # Get Actions from Local Actor
        actions_estimates = self.local_actor(states)
        # Get Value Estimates from Target Critic (without updating it)
        self.local_critic.eval()
        state_values = -self.local_critic(states, actions_estimates).mean()
        self.local_critic.train()
        # Backpropagate and Optimize (only local actor parameters)
        self.actor_optimizer.zero_grad()
        state_values.backward()
        self.actor_optimizer.step()

        # ------ Soft Update Target Networks ------ #
        _soft_update(self.local_critic, self.target_critic)
        _soft_update(self.local_actor, self.target_actor)

    def _reset(self):
        self.local_actor._reset_parameters()
        self.target_actor.load_state_dict(self.local_actor.state_dict())

        self.local_critic._reset_parameters()
        self.target_critic.load_state_dict(self.local_critic.state_dict())

    def checkpoint(self, i_episode, checkpoint_actor='checkpoint_actor.pth', checkpoint_critic='checkpoint_critic.pth'):
        torch.save(self.local_actor.state_dict(), checkpoint_actor.split('.')[0] + f'_episode_{i_episode}.' + checkpoint_actor.split('.')[-1])
        torch.save(self.local_critic.state_dict(), checkpoint_critic.split('.')[0] + f'_episode_{i_episode}.' + checkpoint_critic.split('.')[-1])

    def load_networks(self, checkpoint_actor='checkpoint_actor.pth', checkpoint_critic='checkpoint_critic.pth'):
        self.local_actor.load_state_dict(torch.load(checkpoint_actor))
        self.local_critic.load_state_dict(torch.load(checkpoint_critic))

    def save_networks(self, checkpoint_actor='checkpoint_actor.pth', checkpoint_critic='checkpoint_critic.pth'):
        torch.save(self.local_actor.state_dict(), checkpoint_actor)
        torch.save(self.local_critic.state_dict(), checkpoint_critic)