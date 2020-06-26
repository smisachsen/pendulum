from run_agent_utils import *
from custom_environments import PendulumEnvironment

num_episodes = 500

env = PendulumEnvironment(dt = 0.01, num_actuations = 1,
    max_episode_timesteps = 200)

agent_json_list = ["agents/ppo.json"]

run_agents(agent_json_list, num_episodes, environment = env)
