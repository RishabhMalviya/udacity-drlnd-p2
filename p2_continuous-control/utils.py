from collections import deque

import numpy as np
import matplotlib.pyplot as plt


def _check_solved(i_episode, scores_window, target=30.0) -> bool:
    if i_episode >= 100 and i_episode % 10 == 0:
        print(f'\rEpisode {i_episode}\tAverage Score (over past 100 episodes): {np.mean(scores_window):.2f}')
    
    if np.mean(scores_window)>=target:
        print(f'Environment solved in {i_episode-100} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
        return True

    return False


def plot_scores(scores, save_filename='scores.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')

    plt.savefig(save_filename)


def train_agent(
    env,
    agent,
    n_episodes=250,
    max_t=100,
    save_location='checkpoint.pth'
):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """

    solved = False

    brain_name = env.brain_names[0]

    scores = []
    scores_window = deque(maxlen=100)
    
    try:
        for i_episode in range(1, n_episodes+1):
            # Reset the environment
            env_info = env.reset(train_mode=True)[brain_name]
            state = env_info.vector_observations[0]

            # Reset the agent
            agent.reset()

            # Run the episode
            score = 0            
            for t in range(max_t):
                action = agent.act(state)
                
                env_info = env.step(action)[brain_name]
                next_state, reward, done = env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]
                agent.rewards.append(reward)

                state = next_state
                score += reward

                if done:
                    agent.learn()
                    break

            scores_window.append(score)
            scores.append(score)

            print(f'\rEpisode {i_episode}\t Score: {score:.2f}', end="",flush=True)
            if _check_solved(i_episode, scores_window):
                agent.save(save_location)
                solved = True
                print('\r' \
                'Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                break
    finally:
        env.close()

    return scores, solved


def watch_agent(
    env,
    agent,
    n_episodes=5,
    max_t=100,
    weights_location='checkpoint.pth'
):
    brain_name = env.brain_names[0]

    agent.load_weights(weights_location)

    try:
        for i_episode in range(1, n_episodes+1):
            # Reset the environment
            env_info = env.reset(train_mode=False)[brain_name]
            state = env_info.vector_observations[0]

            # Reset the agent
            agent.reset()

            # Run the episode
            score = 0
            for _ in range(max_t):
                action = agent.act(state)
                
                env_info = env.step(action)[brain_name]
                next_state, reward, done = env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]

                state = next_state
                score += reward

                if done:
                    break 

            print(f'Episode {i_episode}\tScore: {score:.2f}')
    finally:
        env.close()