import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

from agent import Agent
from utils import ScoreKeeper


def train_agent(
        max_t=1000,
        n_episodes=500,
        checkpoint_every=100
):
    # ------ Hyperparameters ------ #
    MAX_T = max_t
    N_EPISODES = n_episodes
    CHECKPOINT_EVERY = checkpoint_every
    NUM_AGENTS = 20

    # ------ Instantiations ------ #
    # Environment
    env = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86_64', no_graphics=True)
    brain_name = env.brain_names[0]
    # Agent
    agent = Agent()
    # Scorekeeper
    scorekeeper = ScoreKeeper()
    # Solved State
    solved = False

    for i_episode in range(1, N_EPISODES+1):
        # ------ Resets ------
        # Environment
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        # Scorekeeper
        scorekeeper.reset()

        # ------ Collect Episode ------ #
        for t in range(MAX_T):
            # Take Action
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state, reward, done = env_info.vector_observations, env_info.rewards, env_info.local_done
            
            # Update Environment and Agent
            for i in range(NUM_AGENTS):
                agent.step(state[i], action[i], reward[i], next_state[i], done[i])
            state = next_state
            
            # Monitoring
            scorekeeper.update_timestep(reward)

            if np.any(done):
                break
        
        # ------ Monitoring and Checkpointing ------ #
        # Monitoring
        solved = scorekeeper.update_episode(i_episode)
        if solved:
            agent.save_networks()
            break
        # Checkpointing
        if i_episode % CHECKPOINT_EVERY== 0:
            agent.checkpoint(i_episode)

    env.close()

    return solved, scorekeeper.scores


def plot_scores(scores, save_filename='scores.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')

    plt.savefig(save_filename)


def watch_agent(
        max_t=1000,
        n_episodes=3,
):
    MAX_T = max_t
    N_EPISODES = n_episodes

    env = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86_64', no_graphics=False)
    brain_name = env.brain_names[0]

    agent = Agent()
    agent.load_networks()

    for _ in tqdm(range(1, N_EPISODES+1)):
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations

        for _ in range(MAX_T):
            env_info = env.step(agent.act(state))[brain_name]

            next_state = env_info.vector_observations
            state = next_state

            done = env_info.local_done
            if np.any(done):
                break

    env.close()


if __name__ == '__main__':
    solved, scores = train_agent(n_episodes=300)
    
    plot_scores(scores)
    
    if solved:
        watch_agent(n_episodes=5)
