import numpy as np


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
    
    def reset(self):
        pass
    
    def load_networks(self):
        pass