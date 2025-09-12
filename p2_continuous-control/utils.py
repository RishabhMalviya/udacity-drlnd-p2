from collections import deque

import numpy as np


class ScoreKeeper:
    def __init__(self, num_agents=20, target_score=30.0, window_len=100):
        self.NUM_AGENTS = num_agents
        self.TARGET_SCORE = target_score
        self.WINDOW_LEN = window_len

        self.scores = []
        self.scores_window = deque(maxlen=self.WINDOW_LEN)
    
    def reset(self):
        self.curr_score = np.zeros(self.NUM_AGENTS)
    
    def update_timestep(self, rewards):
        self.curr_score += rewards
    
    def update_episode(self, i_episode):
        score = np.mean(self.curr_score)
        self.scores.append(score)
        self.scores_window.append(score)
        
        self._check_solved(i_episode)
        
        self.reset()
        
    def _check_solved(self, i_episode):
        print(f'\rEpisode {i_episode}\t Score: {self.scores[-1]:.2f}', end='', flush=True)

        if i_episode >= 100 and i_episode % 10 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score (over past 100 episodes): {np.mean(self.scores_window):.2f}')

        if np.mean(self.scores_window)>=self.TARGET_SCORE:
            print(f'Environment solved in {i_episode-self.WINDOW_LEN} episodes!\tAverage Score: {np.mean(self.scores_window):.2f}')
            return True

        return False