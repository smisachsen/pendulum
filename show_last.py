from tensorforce import Runner, Agent, Environment

from conf import agent_name, environment_name, agent_filename, max_episode_timesteps

environment = Environment.create(environment = "gym",
    level = environment_name, visualize = True,
    max_episode_timesteps=max_episode_timesteps)

agent = Agent.load(directory = "models",
    filename = agent_filename)


runner = Runner(agent = agent, environment = environment)
runner.run(num_episodes = 5)
