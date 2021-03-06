"""Batch evaluation of simulators, instead of running notebooks"""
import itertools
import pickle
from time import time

import fire
import numpy as np

from tracer.recover import TrajectoryRecovery
from tracer.simulator import MobilitySimulator


def evaluate_simulation(
        trajectory_recovery,
        towers,
        sampled_aggregated_data,
        sampled_traces,
        accuracy
):
    """Evaluates a simulator"""
    t_start = time()

    #
    # Gets the recovered traces, the accuracy and error for each assignment
    #
    k_analysis = trajectory_recovery.map_traces_analysis(
        sampled_traces,
        accuracy=accuracy,
        mapping_style='accuracy',
        k=2,
        n_jobs=-1,
    )
    print(f'Took {time() - t_start} to map recovered traces to simulated ones')

    overall_accuracy = np.mean([a['new'][1] for a in k_analysis])
    overall_error = np.mean([a['new'][3] for a in k_analysis])
    overall_accuracy_std = np.std([a['new'][1] for a in k_analysis])
    overall_error_std = np.std([a['new'][3] for a in k_analysis])

    print(f'NEW Overall accuracy: {overall_accuracy}')
    print(f'NEW Overall error: {overall_error}')
    print(f'NEW Overall accuracy std: {overall_accuracy_std}')
    print(f'NEW Overall error std: {overall_error_std}')

    overall_accuracy = np.mean([a['old'][1] for a in k_analysis])
    overall_error = np.mean([a['old'][3] for a in k_analysis])
    overall_accuracy_std = np.std([a['old'][1] for a in k_analysis])
    overall_error_std = np.std([a['old'][3] for a in k_analysis])

    print(f'OLD Overall accuracy: {overall_accuracy}')
    print(f'OLD Overall error: {overall_error}')
    print(f'OLD Overall accuracy std: {overall_accuracy_std}')
    print(f'OLD Overall error std: {overall_error_std}')

    #
    # Also append the simulator attributes to store this information for further analysis.
    #
    # The analysis information is stored as a list with the following items:
    #
    # {
    #   analysis: [
    #     map_results,
    #     global_accuracy,
    #     map_accuracy,
    #     map_error,
    #   ],
    #   simulator_towers,
    #   simulator_traces,
    #   simulator_aggregated_data,
    #   recovered_traces,
    #   recovered_costs,
    # ]
    #
    dump = {
        'analysis': k_analysis,
        'simulator_towers': towers,
        'sampled_traces': sampled_traces,
        'sampled_aggregated_data': sampled_aggregated_data,
        'recovered_costs': trajectory_recovery.C,
        'recovered_distribution': trajectory_recovery.L,
        'recovered_traces': trajectory_recovery.S.T,
    }

    pickle.dump(
        dump,
        open('_'.join([
            './evaluations/.tmp_eval_s_{sampling}_acc_{accuracy}.pkl',
        ]), 'wb')
    )


def evaluate_simulations():
    """Evaluates simulations on a grid of parameters for the simulators"""
    # ##########################################################################
    # #################### Edit the following parameters #######################
    # ##########################################################################

    params_number_users = [4**2, 8**2, 12**2, 16**2, 20**2, 24**2]
    params_number_towers = [4**2, 8**2, 12 **
                            2, 16**2, 20**2, 24**2, 28**2, 32**2]
    params_samplings = [1, 2, 3, 4, 8, 16]  # Sampling
    params_accuracy = [1, 2, 4, 8]  # Size of the districts
    params_velocity = [(0.01, 0.01), (0.05, 0.05)]  # Global velocity

    # ##########################################################################
    # ##########################################################################
    # ##########################################################################

    test_id = 0
    for number_users, number_towers, velocity in itertools.product(
        params_number_users,
        params_number_towers,
        params_velocity,
    ):
        test_id += 1
        print(f'#{test_id} - Creating simulation with parameters:')
        print(f'> Users: {number_users}')
        print(f'> Towers: {number_towers}')
        print(f'> Velocity: {velocity}')
        print(f'> Iteration steps: {96}\n')

        simulator = MobilitySimulator(
            number_users=number_users,
            number_towers=number_towers,
            number_cycles=96,
            velocity=velocity,
            wait_time_max=None,
            mobility_model='random_direction',
            verbose=True,
        )

        print('Generating simulator data...')
        simulator.generate()

        for sampling in params_samplings:
            t_0 = time()
            print(f'Evaluating simulation with parameters:')
            print(f'> Sampling: {sampling}')

            # Samples the aggregated data and traces
            sampled_aggregated_data = simulator.aggregated_data[::sampling, :]
            sampled_traces = simulator.traces[:, ::sampling]

            print(
                f'Aggregated data with shape {simulator.aggregated_data.shape} turned into'
                f' {sampled_aggregated_data.shape}'
            )
            print(
                f'Traces with shape {simulator.traces.shape} turned into'
                f' {sampled_traces.shape}'
            )

            # Recover trajectories
            trajectory_recovery = TrajectoryRecovery(
                number_users=number_users,
                towers=simulator.towers,
                aggregated_data=sampled_aggregated_data,
                vel_friction=0.9,
            )

            t_start = time()
            trajectory_recovery.build_distribution_matrix()
            print(
                f'Took {time() - t_start} to create build distribution matrix')

            t_start = time()
            trajectory_recovery.trajectory_recovery_generator()
            print(
                f'Took {time() - t_start} to create recover traces from aggregated data')

            for accuracy in params_accuracy:
                print(f'Mapping recovered traces with parameters:')
                print(f'> Accuracy: {accuracy}\n')
                t_accuracy = time()

                evaluate_simulation(
                    trajectory_recovery,
                    towers=simulator.towers,
                    sampled_aggregated_data=sampled_aggregated_data,
                    sampled_traces=sampled_traces,
                    accuracy=accuracy,
                )

                print(
                    f'Took {time() - t_accuracy} to complete evaluation '
                    f'with accuracy {accuracy}\n\n'
                )

            print(
                f'Took {time() - t_0} to complete all evaluations with sampling {sampling}\n\n')


class Evaluator(object):
    """Evaluates the mobility models"""

    def run(self):
        evaluate_simulations()


if __name__ == '__main__':
    fire.Fire(Evaluator)
