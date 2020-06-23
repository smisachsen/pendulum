from tensorforce import Agent, Environment, Runner

from conf import environment_name, agent_name, agent_filename, max_episode_timesteps


network = [dict(type='dense', activation = "relu", size=512),
           dict(type='dense', activation = "relu", size=512)]

num_parallel = 4
num_episodes = 1000

environments = [Environment.create(environment = "gym",
    level = environment_name, max_episode_timesteps=max_episode_timesteps)
    for _ in range(num_parallel)]



agent = Agent.create(
    # Agent + Environment
    agent=agent_name,

    environment=environments[0],

    # Network
    network=network,

    # Optimization
    batch_size=256,
    learning_rate=3e-3,
    subsampling_fraction=0.2,
    optimization_steps=50,
    discount = 0.999,


    # Reward estimation
    likelihood_ratio_clipping=0.2,
    estimate_terminal=True,
    # Critic
    critic_network=[dict(type='dense', activation = "relu", size=64),
               dict(type='dense', activation = "relu", size=64)]
,
    critic_optimizer=dict(
        type='multi_step', num_steps=5,
        optimizer=dict(type='adam', learning_rate=1e-3)
    ),
    # Regularization
    entropy_regularization=0.01,
    # TensorFlow etc
    parallel_interactions=num_parallel
)

runner = Runner(agent = agent, environments = environments, num_parallel = num_parallel )
runner.run(num_episodes = num_episodes)

agent.save(directory = "models", filename = agent_filename)
