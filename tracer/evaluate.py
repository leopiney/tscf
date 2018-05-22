"""Batch evaluation of simulators, instead of running notebooks"""
import pickle
from time import time

import fire
import numpy as np

from tracer.recover import TrajectoryRecovery
from tracer.simulator import MobilitySimulator, TraceSimulator


def evaluate_simulator(simulator, number_users, number_towers, sigma):
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

    pickle.dump(
        k_analysis,
        open(f'.tmp_eval_u{number_users}_t{number_towers}_s{sigma}.pkl', 'wb')
    )


def evaluate_simulations(mobility_model):
    """Evaluates simulations on a grid of parameters for the simulators"""
    number_cycles = 24

    params_number_users = [8, 16, 32, 64, 256, 512]
    params_number_towers = [4**2, 6**2, 10**2, 20**2, 30**2, 40**2]
    params_sigma = [0.00025, 0.0005, 0.005]
    # params_velocity = [(0.005, 0.01), (0.1, 0.2), (0.1, 0.3)]
    params_velocity = [(0.01, 0.01), (0.01, 0.02), (0.02, 0.04)]

    test_id = 0
    for (sigma, velocity) in zip(params_sigma, params_velocity):
        for number_users in params_number_users:
            for number_towers in params_number_towers:
                t_0 = time()
                print(f'{test_id}# Evaluating trace simulator with parameters:')
                print(f'> Users: {number_users}')
                print(f'> Towers: {number_towers}')

                if mobility_model == 'custom':
                    print(f'> Sigma: {sigma}\n')
                    simulator = TraceSimulator(
                        number_users=number_users,
                        number_towers=number_towers,
                        number_cycles=number_cycles,
                        sigma=sigma,
                        verbose=True,
                    )
                else:
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
                    sigma=sigma
                )
                test_id += 1

                print(f'Took {time() - t_0} to complete evaluation\n\n')


class Evaluator(object):
    """Evaluates the mobility models"""

    def run(self, mobility_model='custom'):
        evaluate_simulations(mobility_model)


if __name__ == '__main__':
    fire.Fire(Evaluator)
