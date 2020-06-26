from custom_environments import PendulumEnvironment

from tensorforce import Agent

environment = PendulumEnvironment(dt = 0.01, num_actuations = 5, max_episode_timesteps = 200)

agent = Agent.create(agent='agents/ppo.json', environment=environment)

# Train for 200 episodes
for _ in range(200):
    states = environment.reset()
    terminal = False
    while not terminal:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)
