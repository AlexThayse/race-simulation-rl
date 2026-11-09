"""
author:
André Thomaser

refactored by:
Alexander Heilmeier

date:
01.11.2020

.. description::
The functions in this file are related to the reinforcement learning (RL) approach taken to train a Virtual Strategy
Engineer (VSE) within the race simulation to make reasonable strategy decisions. The main script is located in
main_train_rl_agent_dqn.py.
"""

from machine_learning_rl_training.src.rl_environment_single_agent import RaceSimulation
import helper_funcs.src.progressbar
import numpy as np
import matplotlib.pyplot as plt
from machine_learning.src.model_inference import ModelInterpreter


def print_returns_positions(py_env: RaceSimulation,
                            num_races: int,
                            model_path: str,
                            vse_others: str = None):
    """
    Calculate returns and final positions with a PyTorch or TFLite q-network over num_races episodes
    and distinguishes the returns and positions of races with FCY phases and races without FCY phases.

    vse_others: set VSE type of other drivers -> "supervised" or "reinforcement" or "basestrategy" or "realstrategy"
    """

    if vse_others is not None:
        py_env.vse_type = vse_others
    else:
        vse_others = py_env.vse_type

    # initialize q-network
    q_net_lite = {"interpreter": ModelInterpreter(model_path=model_path)}
    q_net_lite["interpreter"].allocate_tensors()
    q_net_lite["input_index"] = q_net_lite["interpreter"].get_input_details()[0]['index']
    q_net_lite["output_index"] = q_net_lite["interpreter"].get_output_details()[0]['index']

    fcy_returns = []
    nofcy_returns = []
    fcy_final_positions = []
    nofcy_final_positions = []

    print('INFO: Evaluating reinforcement VSE by average returns and positions over %i races against %s VSE...'
          % (num_races, vse_others))

    all_returns = []
    all_positions = []
    all_race_times = []

    for i in range(num_races):
        time_step = py_env.reset()
        episode_return = 0.0

        while not time_step.is_last():
            # set NN input
            q_net_lite["interpreter"].set_tensor(
                q_net_lite["input_index"], np.expand_dims(time_step.observation, axis=0))

            # invoke NN
            q_net_lite["interpreter"].invoke()

            # fetch NN output
            action_q = q_net_lite["interpreter"].get_tensor(q_net_lite["output_index"])[0].argmax()

            time_step = py_env.step(action_q)
            episode_return += time_step.reward

        final_position = py_env.race.positions[py_env.race.get_last_compl_lap(py_env.idx_driver), py_env.idx_driver]
        all_returns.append(float(episode_return))
        all_positions.append((float(final_position)))

        completed_lap = py_env.race.get_last_compl_lap(idx=py_env.idx_driver)
        race_time = float(py_env.race.racetimes[completed_lap, py_env.idx_driver])
        all_race_times.append(race_time)

        if not py_env.race.fcy_data['phases']:
            nofcy_returns.append(episode_return)
            nofcy_final_positions.append(final_position)
        else:
            fcy_returns.append(episode_return)
            fcy_final_positions.append(final_position)

        helper_funcs.src.progressbar.progressbar(i=i + 1, i_total=num_races, prefix='INFO: Progress:')

    print('RESULT: Average return (total): %.3f (FCY: %.3f, no FCY: %.3f),'
          ' average position (total): %.1f (FCY: %.1f, no FCY: %.1f),'
          ' FCY races: %i, no FCY races: %i'
          % (float(np.mean(fcy_returns + nofcy_returns)),
             float(np.mean(fcy_returns)),
             float(np.mean(nofcy_returns)),
             float(np.mean(fcy_final_positions + nofcy_final_positions)),
             float(np.mean(fcy_final_positions)),
             float(np.mean(nofcy_final_positions)),
             len(fcy_returns),
             len(nofcy_returns)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(all_returns, bins=30)
    axes[0].set_title("Returns per race")
    axes[0].set_xlabel("Return")
    axes[0].set_ylabel("Number of races")

    axes[1].hist(all_positions, bins=range(1, py_env.race.no_drivers + 2))
    axes[1].set_title("Positions finales")
    axes[1].set_xlabel("Position")

    axes[2].hist(all_race_times, bins=30)
    axes[2].set_title("Time per race")
    axes[2].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig("postproc_results.png", dpi=150)
    plt.close(fig)
