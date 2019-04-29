import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns


class GridWorldPlotter(object):
    """Methods for plotting value functions, uncertainty, etc on a gridworld maze.
    """
    def __init__(self, agent):
        self.agent = agent
        self.env = agent.env

    def plot_maze(self, ax, show_start=True, show_goal=True, show_state_idx=False):
        for idx in range(self.env.num_cols * self.env.num_rows):
            x, y = self.env.get_state_position(idx)
            if self.env.matrix_MDP[x][y] == -1:
                plt.gca().add_patch(
                    patches.Rectangle(
                        (y, self.env.num_rows - x - 1),  #
                        1.0,  # width
                        1.0,  # height
                        facecolor="gray"
                    )
                )
            else:
                pass

        for i in range(self.env.num_cols):
            plt.axvline(i, color='k', linestyle=':')
        plt.axvline(self.env.num_rows, color='k', linestyle=':')

        for j in range(self.env.num_rows):
            plt.axhline(j, color='k', linestyle=':')
        plt.axhline(self.env.num_rows, color='k', linestyle=':')

        plt.xlim([0, 12])
        plt.ylim([0, 12])
        plt.box(False)

        aspect_ratio = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
        ax.set_aspect(aspect_ratio)

        if show_goal:
            plt.text(self.env.goal_x + .1, self.env.goal_y + .1, 'G', fontsize=20, color='green')
        if show_start:
            plt.text(self.env.start_x + .1, self.env.start_y + .1, 'S', fontsize=20, color='black')
        if show_state_idx:
            for idx in self.env.state_indices:
                x, y = self.env.get_state_position(idx)
                plt.text(x+.1, y+.1, str(idx))

    def plot_value(self, ax):

        # Get values
        V = np.zeros(self.env.num_cols * self.env.num_rows)
        for i in self.env.state_indices:
            V[i] = np.dot(self.agent.get_feature_representation(i), self.agent.theta)
        V = V.reshape(self.env.num_rows, self.env.num_cols)
        V[self.env.matrix_MDP == -1] = np.nan  # Assign NaN to wall states

        # plot
        hm_ax = sns.heatmap(V, ax=ax, cmap='viridis')
        hm_ax.invert_yaxis()

    def plot_uncertainty(self, ax):

        # Get uncertainty values (note: change this for non tabular settings)
        uncertainty = np.diag(self.agent.covariance).copy()
        goal_state = self.env.get_state_idx(self.env.goal_x, self.env.goal_y)
        uncertainty[goal_state] = np.nan
        uncertainty = uncertainty.reshape(self.env.num_rows, self.env.num_cols)
        uncertainty[self.env.matrix_MDP == -1] = np.nan  # Assign NaN values to wall states
        # plot
        hm_ax = sns.heatmap(uncertainty, ax=ax)
        hm_ax.invert_yaxis()

if __name__ == '__main__':
    from agent import KTDV
    from environment import GridWorld

    a = KTDV(environment=GridWorld('10x10'))
    p = GridWorldPlotter(a)

    fig, ax = plt.subplots()
    p.plot_maze(ax, show_state_idx=True)
    p.plot_uncertainty(ax)

    plt.show()