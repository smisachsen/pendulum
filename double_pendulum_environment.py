from tensorforce import Environment

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class DoublePendulumEnvironment(Environment):
    def __init__(self, dt = 0.1, num_actuations = 1, max_episode_timesteps = 200):
        super().__init__()
        self.dt = dt
        self.num_actuations = num_actuations
        self.max_episode_timesteps_value = max_episode_timesteps

        self._setup()
        self.reset()

    def _setup(self):
        self.states_list = list()
        self.all_states_list = list()
        self.action_list = list()

        self.G = 9.8  # acceleration due to gravity, in m/s^2
        self.L1 = 1.0  # length of pendulum 1 in m
        self.L2 = 1.0  # length of pendulum 2 in m
        self.M1 = 1.0  # mass of pendulum 1 in kg
        self.M2 = 1.0  # mass of pendulum 2 in kg

    def max_episode_timesteps(self):
        return self.max_episode_timesteps_value

    def states(self):
        return dict(type = "float", shape = (4,))

    def actions(self):
        return dict(type = "int", num_value = 3)

    def reset(self):
        self.time_step = 0

        init_pos = np.random.uniform(low = -np.pi, high = np.pi, size = (2))
        init_vel = np.random.uniform(low = -np.pi/2, high = np.pi/2, size =(2))

        self.state = np.array([init_pos[0], init_vel[0],
            init_pos[1], init_vel[1]])

        self.state = np.radians([120.0, 0.0 ,-10.0, 0.0])

        return self.state

    def get_dydx(self, state, action):
        dydx = np.zeros_like(state)
        dydx[0] = state[1] + action*self.G

        delta = state[2] - state[0]
        den1 = (self.M1+self.M2) * self.L1 - self.M2 * self.L1 * np.cos(delta) * np.cos(delta)
        dydx[1] = ((self.M2 * self.L1 * state[1] * state[1] * np.sin(delta) * np.cos(delta)
                    + self.M2 * self.G * np.sin(state[2]) * np.cos(delta)
                    + self.M2 * self.L2 * state[3] * state[3] * np.sin(delta)
                    - (self.M1+self.M2) * self.G * np.sin(state[0]))
                   / den1)

        dydx[2] = state[3]

        den2 = (self.L2/self.L1) * den1
        dydx[3] = ((- self.M2 * self.L2 * state[3] * state[3] * np.sin(delta) * np.cos(delta)
                    + (self.M1+self.M2) * self.G * np.sin(state[0]) * np.cos(delta)
                    - (self.M1+self.M2) * self.L1 * state[1] * state[1] * np.sin(delta)
                    - (self.M1+self.M2) * self.G * np.sin(state[2]))
                   / den2)

        return dydx

    def execute(self, actions):
        self.time_step += 1
        g = 9.8

        tmp_state = self.state
        action = actions[0]-1

        for _ in range(self.num_actuations):
            dydx = self.get_dydx(tmp_state, action)

            tmp_state += dydx*self.dt
            self.all_states_list.append(list(tmp_state))

        new_state = tmp_state
        self.state = new_state

        self.states_list.append(list(self.state))

        terminal = self.time_step == self.max_episode_timesteps()

        return new_state, terminal, 1

    def plot(self):
        plot_double_pendulum(states = np.array(self.all_states_list),
            L1 = self.L1, L2 = self.L2, dt = self.dt*self.num_actuations)


def plot_double_pendulum(states, L1, L2, dt):
    # integrate your ODE using scipy.integrate.
    y = states


    x1 = L1*np.sin(y[:, 0])
    y1 = -L1*np.cos(y[:, 0])


    x2 = L2*np.sin(y[:, 2]) + x1
    y2 = -L2*np.cos(y[:, 2]) + y1

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text


    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]

        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i*dt))
        return line, time_text


    ani = animation.FuncAnimation(fig, animate, range(1, len(y)),
                                  interval=1000*dt, blit=True, init_func=init)
    plt.show()
