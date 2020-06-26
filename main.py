from tensorforce import Agent, Environment, Runner
from conf import environment_name, agent_name, agent_filename, max_episode_timesteps


network = [dict(type='dense', size=512),
           dict(type='dense', size=512)]

num_parallel = 5
num_episodes = 500

environments = [Environment.create(environment = "gym",
    level = environment_name, max_episode_timesteps=max_episode_timesteps)
    for _ in range(num_parallel)]

env = environments[0]

agent = Agent.create("agents/ppo.json", environment = env)



runner = Runner(agent = agent, environments = environments, num_parallel = num_parallel )
runner.run(num_episodes = num_episodes)

agent.save(directory = "models", filename = agent_filename)
