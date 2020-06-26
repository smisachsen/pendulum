#!/usr/bin/env python
# coding: utf-8

# In[1]:


from custom_environments import PendulumEnvironment
from tensorforce import Agent, Runner


#COOOOOOONFIG
num_train_episodes = 2000
num_parallel = 5

envs = [PendulumEnvironment(dt = 0.01, num_actuations = 10, max_episode_timesteps = 500)
    for _ in range(num_parallel)]


agent = Agent.create("agents/ppo.json", environment = envs[0])



runner = Runner(agent = agent, environments = envs, num_parallel = num_parallel)
runner.run(num_episodes = num_train_episodes)


agent.save(directory = "models", filename = "ppo_custom_pendulum")
