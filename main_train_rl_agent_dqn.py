"""Train the race-strategy DQN using PyTorch (single or shared multi-agent policy)."""

import os
import pickle
import numpy as np
from machine_learning.src.q_network import save_q_network
import matplotlib.pyplot as plt
import racesim
import machine_learning_rl_training
from machine_learning_rl_training.src.dqn import train


# environment parameters
race = "Spa_2019"  # set race (see racesim/input/parameters for possible races)
# VSE type for other drivers: 'basestrategy', 'realstrategy', 'supervised', 'reinforcement' (if already available),
# 'multi_agent' (if VSE should learn for all drivers at once)
controlled_driver = "HAM"
vse_others = "supervised"
mcs_pars_file = "pars_mcs.ini"  # parameter file for Monte Carlo parameters

# hyperparameters
num_iterations = 250_000
replay_buffer_max_length = 200_000
initial_collect_steps = 200
collect_steps_per_iteration = 1

fc_layer_params = (64, 64,)
batch_size = 64
learning_rate = 1e-3
gamma = 1.0  # discount rate
n_step_update = 1
target_update_period = 1
dueling_q_net = False

# training options
log_interval = 100_000
eval_interval = 50_000
num_eval_episodes = 100

# postprocessing (currently not implemented for multi-agent environment)
calculate_final_positions = True  # activate or deactivate evaluation after training
num_races_postproc = 10_000
# VSE type for other drivers: 'basestrategy', 'realstrategy', 'supervised', 'reinforcement' (if already available)
vse_others_postproc = "supervised"


def main():
    evaluate_positions = calculate_final_positions and vse_others != 'multi_agent'
    if calculate_final_positions and not evaluate_positions:
        print('WARNING: Final-position evaluation is not implemented for multi-agent training.')

    race_pars_file = 'pars_%s.ini' % race
    pars_in = racesim.src.import_pars.import_pars(
        use_print=False, use_vse=False, race_pars_file=race_pars_file, mcs_pars_file=mcs_pars_file)[0]
    for driver in pars_in['driver_pars'].values():
        if any(strategy[1] in ['I', 'W'] for strategy in driver['strategy_info']):
            raise RuntimeError('Cannot train for a (partly) wet race: %s' % race)

    env_kwargs = dict(race_pars_file=race_pars_file, mcs_pars_file=mcs_pars_file,
                      use_prob_infl=True, create_rand_events=True)
    if vse_others == 'multi_agent':
        environment_class = machine_learning_rl_training.src.rl_environment_multi_agent.RaceSimulation
    else:
        environment_class = machine_learning_rl_training.src.rl_environment_single_agent.RaceSimulation
        env_kwargs.update(vse_type=vse_others, controlled_driver=controlled_driver)
    train_env = environment_class(**env_kwargs)
    eval_env = environment_class(**env_kwargs)
    print('INFO: Training PyTorch DQN on %s against %s.' % (race, vse_others))
    network, eval_steps, eval_returns, losses = train(
        train_env, eval_env, num_iterations=num_iterations,
        replay_buffer_max_length=replay_buffer_max_length, initial_collect_steps=initial_collect_steps,
        collect_steps_per_iteration=collect_steps_per_iteration, fc_layer_params=fc_layer_params,
        batch_size=batch_size, learning_rate=learning_rate, gamma=gamma,
        n_step_update=n_step_update, target_update_period=target_update_period,
        eval_interval=eval_interval, num_eval_episodes=num_eval_episodes, log_interval=log_interval)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'machine_learning_rl_training', 'output')
    os.makedirs(output_path, exist_ok=True)
    preprocessor_path = os.path.join(output_path, 'preprocessor_reinforcement_%s.pkl' % race)
    model_path = os.path.join(output_path, 'nn_reinforcement_%s.pt' % race)
    with open(preprocessor_path, 'wb') as fh:
        pickle.dump(train_env.cat_preprocessor, fh)
    save_q_network(network, model_path)
    print('RESULT: PyTorch q-network and preprocessor saved in %s.' % output_path)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(eval_steps, eval_returns, marker='o')
    axes[0].set_title('Average return during evaluation')
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('Average return')
    axes[1].plot(np.arange(1, len(losses) + 1), losses)
    axes[1].set_title('Training loss of the DQN per iteration')
    axes[1].set_xlabel('Iterations')
    axes[1].set_ylabel('Loss')
    for axis in axes:
        axis.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, 'training_curves_%s.png' % race), dpi=150)
    plt.close(fig)

    if evaluate_positions:
        py_env = machine_learning_rl_training.src.rl_environment_single_agent.RaceSimulation(
            race_pars_file=race_pars_file, mcs_pars_file=mcs_pars_file, vse_type=vse_others_postproc,
            use_prob_infl=True, create_rand_events=True, controlled_driver=controlled_driver)
        machine_learning_rl_training.src.rl_evaluate_policy.print_returns_positions(
            py_env=py_env, num_races=num_races_postproc, model_path=model_path,
            vse_others=vse_others_postproc)


if __name__ == '__main__':
    main()
