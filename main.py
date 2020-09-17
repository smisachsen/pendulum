from single_pendulum.environment import SinglePendulumContinous
from tensorforce import Agent, Runner
import os
import datetime
import numpy as np
import json
import sys
import argparse

from utils.save_utils import get_new_folder

start_time = datetime.datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument("--num_episodes", type = int, required = True)
parser.add_argument("--num_parallel", type = int, required = True)

args = parser.parse_args()

num_episodes = args.num_episodes
num_parallel = args.num_parallel

network = network = [dict(type='dense', size=512), dict(type='dense', size=512)]

env = SinglePendulumContinous(pendulum_force = 0.1)
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



#agent_path = "agents/ppo.json"
agent_savename = "ppo"
foldername = "single_pendulum_results"
new_folder_path = get_new_folder(foldername)


#agent = Agent.create(agent = agent_path, environment = env)

if num_parallel > 1:
    envs = [SinglePendulumContinous(pendulum_force = 0.1) for _ in range(num_parallel)]
    runner = Runner(agent=agent, environments=envs, num_parallel=num_parallel)
    runner.run(num_episodes = num_episodes)
else:
    runner = Runner(agent=agent, environment=env)
    runner.run(num_episodes = num_episodes)


agent.parallel_interactions = 1
env = env if num_parallel == 1 else envs[0]
eval_runner = Runner(agent=agent, environment=env)
eval_runner.run(num_episodes=1, evaluation = True)



end_time = datetime.datetime.now()



run_info_txt_path = os.path.join(new_folder_path, "run_info.txt")
results_csv_path = os.path.join(new_folder_path, "results.csv")
run_config_json_path = os.path.join(new_folder_path, "run_parameters.json")
agent_savefolder = os.path.join(new_folder_path, "tensorforce_agent")

#save basic run info
with open(run_info_txt_path, "w") as file:
    file.write(f"start time: {start_time} \n")
    file.write(f"end time: {end_time} \n")
    file.write(f"total runtime: {end_time - start_time} \n")

#save results
env.save_data_to_csv(results_csv_path)

#save run info
with open(agent_path, "r") as f:
  data = json.load(f)

for key, val in env.get_run_data().items():
    data[key] = val

with open(run_config_json_path, "w") as file:
    json.dump(data, file)

agent.save(directory = agent_savefolder, filename = agent_savename)
