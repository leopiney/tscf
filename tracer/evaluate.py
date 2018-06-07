"""Batch evaluation of simulators, instead of running notebooks"""
import pickle
from time import time

import fire
import numpy as np

from tracer.recover import TrajectoryRecovery
from tracer.simulator import MobilitySimulator


def evaluate_simulator(simulator, number_users, number_towers, velocity, mobility_model):
    """Evaluates a simulator"""
    # Generate traces
    simulator.generate()

    # Recover trajectories
    trajectory_recovery = TrajectoryRecovery(
        number_users=number_users,
        towers=simulator.towers,
        aggregated_data=simulator.aggregated_data,
        vel_friction=0.9
    )

    t = time()
    trajectory_recovery.build_distribution_matrix()
    print(f'Took {time() - t} to create build distribution matrix')

    t = time()
    trajectory_recovery.trajectory_recovery_generator()
    print(f'Took {time() - t} to create recover traces from aggregated data')

    t = time()

    #
    # Gets the recovered traces, the accuracy and error for each assignment
    #
    k_analysis = trajectory_recovery.map_traces_analysis(
        simulator.traces, mapping_style='accuracy', n_jobs=-1)
    print(f'Took {time() - t} to map traces recovered traces to real ones')

    overall_accuracy = np.mean([a[1] for a in k_analysis])
    overall_error = np.mean([a[3] for a in k_analysis])
    overall_accuracy_std = np.std([a[1] for a in k_analysis])
    overall_error_std = np.std([a[3] for a in k_analysis])

    print(f'Overall accuracy: {overall_accuracy}')
    print(f'Overall error: {overall_error}')
    print(f'Overall accuracy std: {overall_accuracy_std}')
    print(f'Overall error std: {overall_error_std}')

    #
    # Also append the simulator attributes to store this information for further analysis.
    #
    # The analysis information is stored as a list with the following items:
    #
    # [
    #   map_results,
    #   global_accuracy,
    #   map_accuracy,
    #   map_error,
    #   towers,
    #   traces,
    #   aggregated_data,
    # ]
    #
    k_analysis.append(simulator.towers)
    k_analysis.append(simulator.traces)
    k_analysis.append(simulator.aggregated_data)

    pickle.dump(
        k_analysis,
        open('_'.join([
            './evaluate/.tmp_eval',
            f'm_{mobility_model}',
            f'u_{number_users}',
            f't_{number_towers}',
            f'v_{velocity}.pkl',
        ]), 'wb')
    )


def evaluate_simulations(mobility_model):
    """Evaluates simulations on a grid of parameters for the simulators"""
    number_cycles = 24

    params_number_users = [x**2 for x in (4, 8, 12, 16, 20, 24)]
    params_number_towers = [x**2 for x in (4, 8, 12, 16, 20, 24, 28, 32)]
    params_velocity = [(0.01, 0.01), (0.05, 0.05), (0.1, 0.1)]

    test_id = 0
    for velocity in params_velocity:
        for number_users in params_number_users:
            for number_towers in params_number_towers:
                t_0 = time()
                print(f'{test_id}# Evaluating trace simulator with parameters:')
                print(f'> Users: {number_users}')
                print(f'> Towers: {number_towers}')
                print(f'> Velocity: {velocity}\n')

                simulator = MobilitySimulator(
                    number_users=number_users,
                    number_towers=number_towers,
                    number_cycles=number_cycles,
                    velocity=velocity,
                    wait_time_max=None,
                    mobility_model=mobility_model,
                    verbose=True,
                )

                evaluate_simulator(
                    simulator,
                    number_users=number_users,
                    number_towers=number_towers,
                    velocity=velocity,
                    mobility_model=mobility_model,
                )
                test_id += 1

                print(f'Took {time() - t_0} to complete evaluation\n\n')


class Evaluator(object):
    """Evaluates the mobility models"""

    def run(self, mobility_model='custom'):
        evaluate_simulations(mobility_model)


if __name__ == '__main__':
    fire.Fire(Evaluator)
