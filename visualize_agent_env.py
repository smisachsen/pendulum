import pandas as pd
import sys

from tensorforce import Agent, Runner

from single_pendulum_environment import SinglePendulumDiscrete
from run_all_agents import dt, num_actuations, max_episode_timesteps, pendulum_force, env

filename = sys.argv[1]
agent = Agent.load(directory = "models",
    filename = filename)

runner = Runner(agent = agent, environment = env, evaluation=True)

num_episodes_visualize = int(sys.argv[2])
for _ in range(num_episodes_visualize):
    runner.run(num_episodes = 1)
    env.plot()

    s = pd.Series(env.actions_list)
    print(s.describe())
