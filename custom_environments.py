from tensorforce.environments import Environment

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils import plot_single_pendulum

class PendulumEnvironment(Environment):

    def __init__(self, dt = 0.1, num_actuations = 1, max_episode_timesteps = 200):
        super().__init__()

        # position, velocity in radians
        self.reset() #sets self.time_step = 0, and initial conditions (random)

        self.thetas = [self.state[0]]
        self.actions_list = list()
        self.dt = dt
        self.num_actuations = num_actuations
        self.max_episode_timesteps_value = max_episode_timesteps


    def states(self):
        return dict(type = "float", shape = (2,))

    def actions(self):
        return dict(type = "float", shape = (1,),
            min_value = -1.0, max_value = 1.0)

    def max_episode_timesteps(self):
        return self.max_episode_timesteps_value

    def reset(self):
        self.time_step = 0

        init_pos = np.random.uniform(low = -np.pi, high = np.pi, size = 1)[0]
        init_vel = np.random.uniform(low = -np.pi/2, high = np.pi/2, size = 1)[0]
        self.state = np.array([init_pos, init_vel])

        self.rewards = [self.get_reward()]

        return self.state

    def execute(self, actions):
        #update the position and velocity of the pendulum
        self.time_step += 1
        g = 9.8

        tmp_state = self.state

        action = actions[0]
        for _ in range(self.num_actuations):
            pos, vel = tmp_state[0], tmp_state[1]


            acc = -g*np.sin(pos) + 0.8*g*action
            vel += acc*self.dt
            pos += vel*self.dt

            pos = pos%(2*np.pi)

            tmp_state = np.array([pos, vel])

        new_state = tmp_state

        self.state = new_state
        self.thetas.append(pos)
        self.actions_list.append(action)

        #calculate reward given new position
        reward = self.get_reward()
        self.rewards.append(reward)

        #check if last timestep has been reached
        terminal = self.time_step == self.max_episode_timesteps()


        return new_state, terminal, reward

    def get_reward(self):

        reward = 0

        #contribution from height of pendulum outer edge
        pos = self.state[0]
        diff = abs(np.pi - pos)

        reward -= (diff%np.pi)


        return reward

    def plot(self, filename = None):
        anim = plot_single_pendulum(thetas = self.thetas, dt = self.dt, rewards = self.rewards)

        if not filename:
            plt.show()
        else:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            anim.save(filename, writer = writer)










if __name__ == '__main__':
    env = PendulumEnvironment()
