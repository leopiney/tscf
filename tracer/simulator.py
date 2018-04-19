import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from time import time


def distance(p1, p2):
    """Distance between two points"""
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def xamtfos(x, sig):
    aux = (1 / (np.sqrt(2 * np.pi * sig ** 2)))
    return -aux * (np.e ** -(x ** 2 / (2 * (sig ** 2)))) + aux + 1


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class TraceSimulator(object):

    def __init__(
        self,
        number_towers=100,
        number_users=100,
        number_cycles=24,
        method='distance_distribution',
        expander=1,
        sigma=0.03,
        distance_power=5,
        vel_friction=0.9,
        verbose=False,
    ):
        self.number_towers = number_towers
        self.number_users = number_users
        self.number_cycles = number_cycles

        self.method = method

        # for distance_distribution method
        self.expander = expander
        self.sigma = sigma

        # for distance_square method
        self.distance_power = distance_power

        self.vel_friction = vel_friction
        self.verbose = verbose

    def print(self, *args):
        if self.verbose:
            print(*args)

    def generate(self):
        t0 = time()
        self.towers = np.random.rand(self.number_towers, 2)

        self.distances = np.array([
            [
                distance(self.towers[i], self.towers[j])
                for j in range(self.number_towers)
            ]
            for i in range(self.number_towers)
        ])
        self.print(f'Took {time() - t0} to create distrances matrix')

        t = time()
        self.probabilities = self.generate_probabilities()
        self.print(f'Took {time() - t} to create probabilities matrix')

        t = time()
        self.traces = self.generate_weighted_users_traces()
        self.print(f'Took {time() - t} to create user traces')

        t = time()
        self.aggregated_data = self.generate_aggregate_data()
        self.print(f'Took {time() - t} to build aggregated data')

        self.print(f'Took {time() - t0} to generate all')

    def generate_probabilities(self):
        """Generate a matrix of probilities to go from """
        dists = np.copy(self.distances)

        for i in range(self.number_towers):
            for j in range(self.number_towers):
                if self.method == 'distance_distribution':
                    dists[i][j] = (
                        -1 *
                        (dists[i][j] ** 2) *
                        xamtfos(dists[i][j] ** 2, self.sigma) *
                        self.expander
                    )
                elif self.method == 'distance_square':
                    dists[i][j] = -1 * (dists[i][j] + 1) ** self.distance_power

        normalizer = dists.max().max() / 2
        dists -= normalizer

        return np.array([
            softmax(dists[i])
            for i in range(self.number_towers)
        ])

    def is_in_box(self, point):
        return (
            point[0] > 0 and point[0] < 1 and point[1] > 0 and point[1] < 1)

    def get_new_point(self, direction):
        # direction is a list with two elemenst, the vector
        vel_x = direction[1][0] - direction[0][0]
        vel_y = direction[1][1] - direction[0][1]
        x = direction[1][0] + vel_x
        y = direction[1][1] + vel_y

        while not self.is_in_box([x, y]):
            # Tracer()()

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
        distances = [distance(point, x) for x in self.towers]
        return np.argmin(distances)

    def generate_weighted_users_traces(self):
        def generate_weighted_user_trace():
            towers_ids = np.arange(self.number_towers)

            trace = []
            direction = []
            for c in range(self.number_cycles):
                if c == 0:
                    # For the first towers the chance of selecting a tower is equally distributed
                    tower = np.random.choice(towers_ids)
                    trace.append(tower)
                    direction.append(self.towers[tower])
                elif c == 1:
                    last_tower = trace[c - 1]
                    tower = np.random.choice(towers_ids, p=self.probabilities[last_tower])
                    trace.append(tower)
                    direction.append(self.towers[tower])
                else:
                    new_point = self.get_new_point(direction)
                    nearest_tower = self.get_nearest_tower(new_point)
                    tower = np.random.choice(towers_ids, p=self.probabilities[nearest_tower])
                    trace.append(tower)
                    direction = [direction[1], self.towers[tower]]

            return trace

        return np.array([
            generate_weighted_user_trace()
            for _ in range(self.number_towers)
        ])

    def generate_aggregate_data(self):
        """Returns how many users were in each step of the cycle based on traces"""
        output = np.zeros((self.number_towers, self.number_cycles))

        for tower in range(self.number_towers):
            for user in range(self.number_users):
                for cycle in range(self.number_cycles):
                    output[tower][cycle] += self.traces[user][cycle] == tower

        return output

    def plot_towers(self, figsize=(8, 8), annotate_towers=True):
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
                ax.annotate(f'    T{i}', (df_towers.iloc[i].x, df_towers.iloc[i].y))

        plt.gca().set_aspect('equal', adjustable='box')
        return ax

    def plot_user_trace(self, user_id=0, figsize=(12, 12), annotate_towers=True, verbose=False):
        ax = self.plot_towers(figsize=figsize, annotate_towers=annotate_towers)

        cycol = itertools.cycle('bgrcmk')

        trace = self.traces[user_id]

        for i in range(self.number_cycles - 1):
            if trace[i] == trace[i + 1]:
                self.print(f'Cycle #{i}: Staying in tower T{trace[i]}')
                continue
            x1, y1 = self.towers[trace[i]]
            x2, y2 = self.towers[trace[i + 1]]

            self.print(
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

        plt.gca().set_aspect('equal', adjustable='box')
