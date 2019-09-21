import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    def __init__(self, problem):
        self.problem = problem

    def visualize_path(self, seq, times):


        n_labels = len(seq)  # Number of labels

        # setup the plot
        fig, ax = plt.subplots(1, 1)

        # define the colormap
        cmap = plt.cm.jet
        # extract all colors from the .jet map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # create the new map
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
        # Bounds for the tick marks on color map
        bounds = np.linspace(0, n_labels, n_labels + 1)

        # Path positions for service ship
        # Path to plot
        x_path = []
        y_path = []

        # Locations of start of full ship paths
        x_first = []
        y_first = []

        # Full ship paths data
        x = []
        y = []
        tags = []  # Tags used for colors

        # Add service ship contact position for each shipZ
        for ship, t in zip(seq, times):

            # Add first and last times for this ship
            s = self.problem.get_ship(ship)
            f, l = s.get_times()
            x_first.append(f)
            y_first.append(l)

            positions = s.get_positions()
            x.extend(positions[0])
            y.extend(positions[1])

            position = s.get_position(time=t)

            n_labels -= 1

            x_path.append(position[0])
            y_path.append(position[1])
            plt.annotate(str(t), xy=(position[0], position[1]), fontsize='large')

        # The paths of the ships with coloring
        full_paths = ax.scatter(x, y, c=tags, cmap=cmap, alpha=0.5)
        plt.scatter(x_first, y_first, c='k', alpha=1, s=5)
        # The path of the service ship with marking
        plt.plot(x_path, y_path, c='red', ls='-', lw=2, marker='*', ms=10)

        # create the colorbar
        cb = plt.colorbar(full_paths, spacing='proportional', ticks=bounds)
        cb.set_label('Ships')
        ax.set_title('Service Ship Path')
        plt.show()
