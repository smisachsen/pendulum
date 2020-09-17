import argparse
import os
import sys
import csv
import socket
import numpy as np

from tensorforce.agents import Agent
from tensorforce.execution import Runner

from RemoteEnvironmentClient import RemoteEnvironmentClient
from single_pendulum.environment import SinglePendulumContinous


ap = argparse.ArgumentParser()
ap.add_argument("-n", "--number-servers", required=True, help="number of servers to spawn", type=int)
ap.add_argument("-p", "--ports-start", required=True, help="the start of the range of ports to use", type=int)
ap.add_argument("-t", "--host", default="None", help="the host; default is local host; string either internet domain or IPv4", type=str)

args = vars(ap.parse_args())

number_servers = args["number_servers"]
ports_start = args["ports_start"]
host = args["host"]

if host == 'None':
    host = socket.gethostname()

example_environment = SinglePendulumContinous(pendulum_force = 0.1)

use_best_model = True

environments = []
for crrt_simu in range(number_servers):
    environments.append(RemoteEnvironmentClient(
        example_environment, verbose=0, port=ports_start + crrt_simu, host=host,
        timing_print=(crrt_simu == 0)
    ))

if use_best_model:
    evaluation_environment = environments.pop()
else:
    evaluation_environment = None

network = [dict(type='dense', size=512), dict(type='dense', size=512)]

agent = Agent.create(
    agent = "ppo",
    environment = env,

    network= network,
    update_frequency = 20,
    batch_size = 20,
    learning_rate = 0.001,
    discount = 0.999,

    critic_network=network,
    critic_optimizer=dict(
        type='multi_step', num_steps=5,
        optimizer=dict(type='adam', learning_rate=1e-3)
    ),

    entropy_regularization=0.01,
    parallel_interactions=num_parallel
)

runner = Runner(
    agent=agent, environments=environments, evaluation_environment=evaluation_environment
)

cwd = os.getcwd()
evaluation_folder = "env_" + str(number_servers - 1)
sys.path.append(cwd + evaluation_folder)
# out_drag_file = open("avg_drag.txt", "w")

runner.run(
    num_episodes=400, sync_episodes=True,
    save_best_agent=use_best_model
)
# out_drag_file.close()
runner.close()