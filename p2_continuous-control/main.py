from unityagents import UnityEnvironment

from agent import Agent
from utils import train_agent, plot_scores, watch_agent


def run():
    # Train Agent
    scores, solved = train_agent(
        env    = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86_64', no_graphics=True),
        agent  = Agent(state_size=33, action_size=4, seed=0),
        n_episodes=300,
        max_t=1000
    )
    plot_scores(scores)

    # Watch Agent
    if solved:
        watch_agent(
            env    = UnityEnvironment(file_name='./Reacher_Linux/Reacher.x86_64'),
            agent  = Agent(state_size=37, action_size=4, seed=0)
        )
