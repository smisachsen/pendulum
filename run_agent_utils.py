import sys

from tensorforce import Runner, Agent

def run_agents(agent_json_list, num_episodes, environment = None, environments = None):
    env = environment if environment else environments[0]
    for agent_json in agent_json_list:
        agent = Agent.create(agent_json, environment = env)
        _run_agent(agent, num_episodes, environment = environment,
            environments = environments)

def _run_agent(agent, num_episodes, environment = None, environments = None):
    if environment is not None:
        runner = Runner(agent = agent, environment = environment)

    elif environments is not None:
        runner = Runner(agent = agent, environments = environments,
            num_parallel = len(environments))

    else:
        print("environment or environments must be != None")
        sys.exit()


    runner.run(num_episodes = num_episodes)
    agent.save(directory = "models", filename = str(agent))
