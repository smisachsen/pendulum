import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
