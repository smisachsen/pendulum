from tensorforce.environments import Environment

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class SinglePendulumBase(Environment):
    """
    Base class used to construct single pendulum with
    continous and discrete action space
    """

    def __init__(self, dt = 0.1, num_actuations = 5, max_episode_timesteps = 200,
        init_position_range = [-np.pi, np.pi],
        init_velocity_range = [-1, 1],
        pendulum_force = 0.5):

        super().__init__()
        self.init_position_range = init_position_range
        self.init_velocity_range = init_velocity_range

        # position, velocity in radians
        self.reset() #sets self.time_step = 0, and initial conditions (random)

        self.thetas = list()
        self.actions_list = list()
        self.theta_dots = list()

        self.dt = dt
        self.num_actuations = num_actuations
        self.max_episode_timesteps_value = max_episode_timesteps
        self.pendulum_force = pendulum_force

    def max_episode_timesteps(self):
        return self.max_episode_timesteps_value

    def states(self):
        return dict(type = "float", shape = (2,))

    def actions(self):
        raise NotImplementedError

    def apply_action(self, action):
        #update the position and velocity of the pendulum
        self.time_step += 1
        g = 9.8

        tmp_state = self.state
        self.thetas.append(tmp_state[0])
        self.theta_dots.append(tmp_state[1])

        for _ in range(self.num_actuations):
            pos, vel = tmp_state[0], tmp_state[1]

            acc = -g*np.sin(pos) + self.pendulum_force*g*action
            vel += acc*self.dt
            pos += vel*self.dt

            pos = pos%(2*np.pi)

            tmp_state = np.array([pos, vel])

        new_state = tmp_state

        self.state = new_state
        self.actions_list.append(action)

        #calculate reward given new position
        reward = self.get_reward()
        self.rewards.append(reward)

        #check if last timestep has been reached
        terminal = self.time_step == self.max_episode_timesteps()


        return new_state, terminal, reward

    def reset(self):
        self.time_step = 0
        low_pos, high_pos = self.init_position_range[0], self.init_position_range[1]
        low_vel, high_vel = self.init_velocity_range[0], self.init_velocity_range[1]

        init_pos = np.random.uniform(low = low_pos, high = high_pos, size = 1)[0]
        init_vel = np.random.uniform(low = low_vel, high = high_vel, size = 1)[0]

        self.state = np.array([init_pos, init_vel])
        self.thetas = list()
        self.actions_list = list()
        self.theta_dots = list()

        self.rewards = []

        return self.state

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

    def save_data_to_csv(self, path):
        df = pd.DataFrame()
        df["actions"] = self.actions_list
        df["thetas"] = self.thetas
        df["theta_dots"] = self.theta_dots
        df["rewards"] = self.rewards

        df.to_csv(path, index = False)

    def get_run_data(self):
        data = dict()
        data["num_episodes"] = self.max_episode_timesteps_value
        data["pendulum_force"] = self.pendulum_force
        data["dt"] = self.dt
        data["num_actuations"] = self.num_actuations

        return data



class SinglePendulumDiscrete(SinglePendulumBase):

    def actions(self):
        return dict(type = "int", num_values = 3)

    def execute(self, actions):
        action = actions-1
        return self.apply_action(action)

class SinglePendulumContinous(SinglePendulumBase):

    def actions(self):
        return dict(type = "float", shape = (1), min_value = -1.0,
            max_value = 1.0)

    def execute(self, actions):
        return self.apply_action(actions[0])


def plot_single_pendulum(thetas, dt, rewards = None):

    x =  np.sin(thetas)
    y = -np.cos(thetas)

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on = False, xlim=(-2, 2), ylim=(-2, 2))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], 'o-', lw=2)
    time_template = 'time = %.1fs, reward = %.3f'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        xx = [0, x[i]]
        yy = [0, y[i]]

        line.set_data(xx, yy)
        time_text.set_text(time_template  % (i*dt, rewards[i]))

        return line, time_text

    ani = animation.FuncAnimation(fig, animate, range(1, len(y)),
                                  interval=dt*1000, blit=True, init_func=init)
    return ani
