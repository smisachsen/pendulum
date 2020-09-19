import socket

from multiprocessing import Process
from tensorforce import Runner, Agent
from single_pendulum.environment import SinglePendulumContinous

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
parser.add_argument("--num-parallel", type = int, required = True)

args = parser.parse_args()

num_episodes = args.num_episodes
num_parallel = args.num_parallel

if __name__ == '__main__':
    try:
        savefolder = "saver_data"
        agent_name = "ppo_agent"

        num_servers = num_parallel
        port_start = 7010
        ports = range(port_start, port_start + num_servers)
        host = socket.gethostname()



        #launch servers
        processes = list()
        for i, port in enumerate(ports):
            verbose = i==0
            env = SinglePendulumContinous()
            proc = Process(target=launch_server, args = (host, port, verbose, env))
            proc.start()
            processes.append(proc)
            time.sleep(1)

        print(f"started {len(processes)} servers")

        envs = []
        example_env = SinglePendulumContinous()
        for i, port in enumerate(ports):
            client = Client(environment=example_env, port=port, host=host, verbose = i==0)
            envs.append(client)

        print("setup clients DONE")

        try:
            agent = Agent.load(directory=savefolder, filename=agent_name, environment=example_env)
            print(f"loaded {agent_name} agent from {savefolder}")
        except:
            print("failed to load agent. Setting manually")

            network = [dict(type='dense', size=512), dict(type='dense', size=512)]
            agent = Agent.create(
                # Agent + Environment
                agent='ppo', environment=example_env,
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
                parallel_interactions=num_servers
                )

        print("setup agent DONE")

        runner = Runner(agent = agent, environments=envs, num_parallel = num_servers)
        print("setup runner DONE")

        try:
            runner.run(num_episodes = num_episodes, sync_episodes=True, save_best_agent=True)
        except:
            print("exception")
            runner.close()

        for env in envs:
            env.socket.close()

    except:
        pass

    finally:
        #save agent

        create_folder_if_not_exists(savefolder)
        agent.save(directory=savefolder, filename = agent_name)
        print(f"saved agent to {savefolder}")
