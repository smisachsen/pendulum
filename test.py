from single_pendulum_environment import *

from tensorforce import Agent
import numpy as np

environment = SinglePendulumDiscrete(dt = 0.01, num_actuations = 5,
    max_episode_timesteps = 500)

agent = Agent.create(agent='agents/deepq.json', environment=environment)

# Train for 200 episodes
rewards = []
for ep in range(1000):
    states = environment.reset()
    terminal = False

    episode_rewards = []
    while not terminal:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        episode_rewards.append(reward)

    print(f"ep: {ep} mean episode reward:", np.mean(episode_rewards))
