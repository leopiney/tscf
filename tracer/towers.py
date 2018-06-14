"""Towers and maps module"""
import itertools
import numpy as np
import pandas as pd


from tracer.utils import is_in_box


class TowersManager(object):
    def __init__(self, towers, vel_friction=0.9):
        self.towers = towers
        self.vel_friction = vel_friction

        self.number_towers = towers.shape[0]
        self.distances = self.generate_distances()

    def generate_distances(self):
        return np.array([
            np.sqrt(
                np.sum(
                    (self.towers - self.towers[i]) ** 2,
                    axis=1
                )
            )
            for i in range(self.number_towers)
        ])

    def get_new_point(self, direction):
        # direction is a list with two elemenst, the vector
        vel_x = direction[1][0] - direction[0][0]
        vel_y = direction[1][1] - direction[0][1]
        x = direction[1][0] + vel_x
        y = direction[1][1] + vel_y

        while not is_in_box([x, y]):
            if x > 1:
                vel_x = - vel_x
                x = 2 * 1 - x
            if x < 0:
                vel_x = - vel_x
                x = 2 * 1 + x
            if y > 1:
                vel_y = - vel_y
                y = 2 * 1 - y
            if y < 0:
                vel_y = - vel_y
                y = 2 * 1 + y

            vel_x *= self.vel_friction
            vel_y *= self.vel_friction

        return [x, y]

    def get_nearest_tower(self, point):
        return np.argmin(
            np.sqrt(
                np.sum(
                    (self.towers - point) ** 2,
                    axis=1
                )
            )
        )

    def plot_towers(self, figsize=(8, 8), annotate_towers=True):
        import matplotlib.pyplot as plt

        df_towers = pd.DataFrame(self.towers, columns=['x', 'y'])
        ax = df_towers.plot.scatter(
            x='x',
            y='y',
            ylim=(0, 1),
            xlim=(0, 1),
            figsize=figsize,
            marker='x'
        )

        if annotate_towers:
            for i in range(len(df_towers)):
                ax.annotate(
                    f'    T{i}', (df_towers.iloc[i].x, df_towers.iloc[i].y))

        plt.gca().set_aspect('equal', adjustable='box')
        return ax

    def plot_user_trace_aux(self, trace, ax, colors='bgrcmk', verbose=False):
        cycol = itertools.cycle(colors)

        for i in range(len(trace) - 1):
            if trace[i] == trace[i + 1]:
                if verbose:
                    print(f'Cycle #{i}: Staying in tower T{trace[i]}')
                continue

            x1, y1 = self.towers[trace[i]]
            x2, y2 = self.towers[trace[i + 1]]

            if verbose:
                print(
                    f'Cycle #{i}: Switching from T{trace[i]} '
                    f'to T{trace[i + 1]}'
                )

            color = next(cycol)
            ax.arrow(
                x1,
                y1,
                x2 - x1,
                y2 - y1,
                head_width=0.01,
                head_length=0.01,
                fc=color,
                ec=color
            )

    def plot_user_trace(
        self,
        trace,
        figsize=(12, 12),
        annotate_towers=True,
        verbose=False
    ):
        import matplotlib.pyplot as plt

        ax = self.plot_towers(
            figsize=figsize,
            annotate_towers=annotate_towers
        )

        self.plot_user_trace_aux(trace, ax, verbose=verbose)

        plt.gca().set_aspect('equal', adjustable='box')

    def plot_user_mapping_trace(
        self,
        recovered_trace,
        real_trace,
        figsize=(12, 12),
        annotate_towers=True,
        verbose=False
    ):
        import matplotlib.pyplot as plt

        ax = self.plot_towers(
            figsize=figsize,
            annotate_towers=annotate_towers
        )

        self.plot_user_trace_aux(recovered_trace, ax, colors='r', verbose=verbose)
        self.plot_user_trace_aux(real_trace, ax, colors='b', verbose=verbose)

        plt.gca().set_aspect('equal', adjustable='box')
