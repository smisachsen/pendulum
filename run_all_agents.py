from optimal_agent_tuning import run_agents
from single_pendulum_environment import *

num_episodes = 500
num_actuations = 5
dt = 0.01
max_episode_timesteps = 400
pendulum_force = 0.5

env = SinglePendulumDiscrete(dt = dt, num_actuations = num_actuations,
    max_episode_timesteps = max_episode_timesteps,
    pendulum_force = pendulum_force)


if __name__ == '__main__':

    agent_json_list = ["agents/deepq.json", "agents/dueling_dqn.json"]
    agent_names = ["deepq_agent1", "dueling_dqn_agent1"]

    run_agents(agent_json_list, num_episodes,
        agent_names = agent_names,  environment = env,
        output = True)
