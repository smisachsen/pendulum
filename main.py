from single_pendulum.environment import SinglePendulumContinous
from tensorforce import Agent, Runner
import os
import datetime
import numpy as np
import json
import sys

from utils.save_utils import get_new_folder

start_time = datetime.datetime.now()

num_episodes = int(sys.argv[1])
agent_path = "agents/ppo.json"
agent_savename = "ppo"

env = SinglePendulumContinous(pendulum_force = 0.1)
agent = Agent.create(agent = agent_path, environment = env)
runner = Runner(agent = agent, environment = env)


runner.run(num_episodes = num_episodes)
runner.run(num_episodes = 1, evaluation = True)


end_time = datetime.datetime.now()

foldername = "single_pendulum_results"
new_folder_path = get_new_folder(foldername)

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
