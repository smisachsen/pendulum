import socket

from multiprocessing import Process
from tensorforce import Runner, Agent
from single_pendulum.environment import SinglePendulumDiscrete

from socket_utils.socket_server import Server
from socket_utils.socket_env import Client
from socket_utils.utils import *
from utils.save_utils import *
import time
import os
import argparse

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

parser = argparse.ArgumentParser()
parser.add_argument("--num-episodes", type = int, required = True)
#parser.add_argument("--num-parallel", type = int, required = True)

args = parser.parse_args()

num_episodes = args.num_episodes
#num_parallel = args.num_parallel


savefolder = "saver_data"
agent_name = "single_pendulum_deepq_agent"
savefolder_agent = os.path.join(savefolder, agent_name)

env = SinglePendulumDiscrete()

try:
    agent = Agent.load(directory=savefolder_agent, filename=agent_name, environment=env)
    print(f"loaded {agent_name} agent from {savefolder}")

except:
    print("failed to load agent. Setting manually")

    network = [dict(type='dense', size=512), dict(type='dense', size=512)]
    agent = Agent.create(
        # Agent + Environment
        agent='dqn',
        environment=env,
        network=network,
        # Optimization
        batch_size=20,
        learning_rate=1e-3,
        memory = 10_000,
        discount = 0.999,
        exploration=dict(
            type='decaying', unit='episodes', decay='exponential',
            initial_value=0.9, decay_steps=1000, decay_rate=0.5 
        )
    )



print("setup agent DONE")

try:
    runner = Runner(agent = agent, environment=env)
    runner.run(num_episodes = num_episodes)
    env.plot()


    create_folder_if_not_exists(savefolder)
    create_folder_if_not_exists(savefolder_agent)

except KeyboardInterrupt:
    print("Quiting program")

finally:
    agent.save(directory=savefolder_agent, filename = agent_name)
    print(f"saved agent to {savefolder}")