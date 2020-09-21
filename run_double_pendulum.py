from double_pendulum.double_pendulum_env2 import CartPoleEnv
from tensorforce import Agent, Runner

import time
import numpy as np
import argparse

import tensorflow as tf
tf.get_logger().setLevel('ERROR')


parser = argparse.ArgumentParser()
parser.add_argument("--num-episodes", type = int, required = True)

args = parser.parse_args()

num_episodes = args.num_episodes



env = CartPoleEnv()
env.reset()

network = [dict(type='dense', size=512), dict(type='dense', size=512)]
agent = Agent.create(
    # Agent + Environment
    agent='ppo', environment=env,
    #max_episode_timesteps=nb_actuations,
    # TODO: nb_actuations could be specified by Environment.max_episode_timesteps() if it makes sense...
    # Network
    network=network,
    # Optimization
    batch_size=20, learning_rate=1e-3, subsampling_fraction=0.2, optimization_steps=25,
    # Reward estimation
    likelihood_ratio_clipping=0.2, estimate_terminal=True,  # ???
    # TODO: gae_lambda=0.97 doesn't currently exist
    # Critic
    critic_network=network,
    critic_optimizer=dict(
        type='multi_step', num_steps=5,
        optimizer=dict(type='adam', learning_rate=1e-3)
    ),
    # Regularization
    entropy_regularization=0.01,
    # TensorFlow etc
    parallel_interactions=1
    )


episode_rewards = []
episode_states = []


for ep in range(num_episodes):
    ep_rew = list()
    ep_states = list()
    states = env.reset()
    done = False
    ts_counter = 0
    while not done:
        ts_counter += 1
        actions = agent.act(states)
        states, reward, done, _ = env.step(actions)
        agent.observe(reward)

        ep_rew.append(reward)
        ep_states.append(states)

    print(f"episode: {ep} mean reward: {np.mean(ep_rew)}. Lasted {ts_counter} rounds before termination")
    episode_rewards.append(ep_rew)
    episode_states.append(ep_states)
