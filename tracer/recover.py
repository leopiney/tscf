"""Trajectory recovery module"""
import itertools
import numpy as np

from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm, tqdm_notebook

from tracer.towers import TowersManager


class TrajectoryRecovery(object):
    """Upturn users trajectories from ash using aggregated data"""

    def __init__(
        self,
        number_users,
        towers,
        aggregated_data,
        vel_friction=0.9
    ):
        self.aggregated_data = aggregated_data
        self.towers = towers
        self.vel_friction = vel_friction

        self.number_users = number_users
        self.number_cycles = aggregated_data.shape[0]
        self.number_towers = aggregated_data.shape[1]

        self.towers_manager = TowersManager(towers, vel_friction=vel_friction)
        self.distances = self.towers_manager.distances

    def build_distribution_matrix(self):
        L = []

        for cycle_counts in self.aggregated_data:
            L.append(
                np.array(list(
                    itertools.chain(*(
                        [tower_index] * int(count)
                        for tower_index, count in enumerate(cycle_counts)
                    ))
                ))
            )

        self.L = np.array(L)

    def trajectory_recovery_generator(self, show_progress=False):
        self.S = []
        self.C = [None]

        progress_indicator = tqdm_notebook if show_progress else tqdm

        for cycle in progress_indicator(range(self.number_cycles), 'Recovering'):
            if cycle == 0:
                self.S.append(np.random.permutation(self.L[0]))
            else:
                if cycle == 1:
                    #
                    # If it's on the night, we estimate the next location as the last one.
                    #
                    L_next_est = self.S[cycle - 1]
                else:
                    #
                    # During daylight, we estimate the next location taking into account
                    # the current users trajectory and direction.
                    #
                    L_next_est = []
                    for user in range(self.number_users):
                        direction = [
                            self.towers[self.S[cycle - 2][user]],
                            self.towers[self.S[cycle - 1][user]]
                        ]
                        new_point = \
                            self.towers_manager.get_new_point(direction)
                        l_next_est = \
                            self.towers_manager.get_nearest_tower(new_point)

                        L_next_est.append(l_next_est)

                    L_next_est = np.array(L_next_est)

                L_next = self.L[cycle]

                #
                # Calculate the cost matrix as the distance between the estimated tower and the
                # rest.
                #
                C_next = np.zeros((self.number_users, self.number_users))

                for i, l_next_est in enumerate(L_next_est):
                    for j, l_next in enumerate(L_next):
                        C_next[i, j] = self.distances[l_next, l_next_est]

                #
                # Append the cost matrix to the collection of cost matrices
                #
                self.C.append(C_next)

                #
                # Ref: https://docs.scipy.org/doc/scipy-0.18.1/reference/
                #  generated/scipy.optimize.linear_sum_assignment.html
                #
                # Solves the assignament problem of assiging the users the next
                # location taking into account the matrix of costs. The result
                # comes in the form of indexes, being the row_index the number of users
                # in this case, and the col_index the tower index. Therefore, the
                # col_index is the S_next we're looking for.
                #
                _, col_ind = linear_sum_assignment(C_next)

                self.S.append(L_next[col_ind])

        self.C = np.array(self.C)
        self.S = np.array(self.S)

        return {
            'recovered_costs': self.C,
            'recovered_trajectories': self.S,
        }

    def get_traces_common_elements(self, trace_1, trace_2):
        return np.sum(trace_1 == trace_2)

    def get_traces_distance_error(self, trace_1, trace_2):
        return np.sum([
            self.distances[z_t, y_t]
            for z_t, y_t in zip(trace_1, trace_2)
        ])

    def map_traces(self, real_traces, mapping_style, show_progress=False):
        """Maps the recovered traces with real ones

        @param real_traces The traces to map with the recovered traces
        @param mapping_style Could be either 'accuracy' or 'error'
        """
        #
        # Build an boolean array where each value represents if that real_trace
        # has already been used
        #
        used_traces = np.array([False for _ in real_traces])

        #
        # Store the accuracy, error and best match for each mapping between the recovered
        # traces (self.S)
        # and the real_traces
        #
        mapping_accuracy = np.zeros(len(self.S.T)).astype('int')
        mapping_error = np.zeros(len(self.S.T))
        result = np.zeros(len(self.S.T)).astype('int')

        #
        # Generate a random index to iterate through the recoevered traces in a random order
        #
        random_index = np.arange(len(self.S.T))
        np.random.shuffle(random_index)

        progress_indicator = tqdm_notebook if show_progress else tqdm

        for recovered_trace_index in progress_indicator(random_index, 'Mapping'):
            recovered_trace = self.S.T[recovered_trace_index]

            #
            # Compare the recovered_trace with the real_traces
            # Compute the accuracy as the number of towers they have in common in the same
            # time slot. The common_elements is an array that shows for each real_trace how
            # many towers they have in common with the recovered_trace
            #
            common_elements = np.array([
                self.get_traces_common_elements(recovered_trace, real_trace)
                for real_trace in real_traces
            ])
            mapping_errors = np.array([
                self.get_traces_distance_error(
                    trace_1=recovered_trace,
                    trace_2=real_trace
                )
                for real_trace in real_traces
            ])

            #
            # Flag all the common_elements values of the real_traces that have been used
            # So that they won't be selected as candidates
            #
            common_elements[used_traces] = -1

            # Select the real_trace that matches the best with the recovered
            if mapping_style == 'accuracy':
                best_match_index = np.argmax(common_elements)
            elif mapping_style == 'error':
                best_match_index = np.argmin(mapping_errors)
            else:
                raise ValueError(
                    f'Invalid mapping style {mapping_style}.'
                    ' Select either "accuracy" or "error"'
                )

            mapping_accuracy[recovered_trace_index] = common_elements[best_match_index]
            mapping_error[recovered_trace_index] = mapping_errors[best_match_index]

            result[recovered_trace_index] = best_match_index

            # Mark best match trace as used
            used_traces[best_match_index] = True

        mapping_accuracy = np.array(mapping_accuracy)

        global_accuracy = np.sum(
            mapping_accuracy / self.number_cycles) / self.number_users
        return result, global_accuracy, mapping_accuracy, mapping_error

    def map_traces_analysis(self, real_traces, mapping_style, k=10, n_jobs=-1):
        """Returns the mapping results starting at a random generated trace

        To avoid falling into local minimum values its appropiate to run
        the algorithm several times starting in different points to see that
        the results converge into a similar value.

        @returns an array of tuples (result, global_accuracy, mapping_accuracy, mapping_error)"""
        return Parallel(n_jobs=n_jobs)(
            delayed(self.map_traces)(real_traces, mapping_style)
            for _ in range(k)
        )
