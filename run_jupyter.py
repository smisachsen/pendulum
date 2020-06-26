
from custom_environments import PendulumEnvironment
from tensorforce import Agent, Runner
import pandas as pd


from conf import agent_name, environment_name, agent_filename, max_episode_timesteps

env = PendulumEnvironment(dt = 0.01, num_actuations = 5, max_episode_timesteps = 200)

filename = "ProximalPolicyOptimization"
agent = Agent.load(directory = "models",
    filename = filename)


runner = Runner(agent = agent, environment = env, evaluation=True)

for _ in range(5):
    runner.run(num_episodes = 1)
    env.plot()

    s = pd.Series(env.actions_list)
    print(s.describe())
