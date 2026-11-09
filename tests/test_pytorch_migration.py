"""Integration checks for the framework migration without a long training run."""

from pathlib import Path
import os
from io import StringIO
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from machine_learning.src.model_inference import ModelInterpreter
from machine_learning.src.q_network import save_q_network
from machine_learning_rl_training.src.dqn import QNetwork, ReplayBuffer, train, update_network
from machine_learning_rl_training.src.rl_environment_single_agent import RaceSimulation
from machine_learning_rl_training.src.rl_environment_multi_agent import RaceSimulation as MultiRaceSimulation


class MigrationTests(unittest.TestCase):
    def test_training_entry_point_and_vse_checkpoint(self):
        import main_train_rl_agent_dqn as entry
        from racesim.src.vse_reinforcement import VSE_REINFORCEMENT
        with tempfile.TemporaryDirectory() as directory:
            previous_directory = os.getcwd()
            try:
                os.chdir(directory)
                with patch.multiple(entry, __file__=str(Path(directory) / 'main_train_rl_agent_dqn.py'),
                                    num_iterations=2, initial_collect_steps=2, batch_size=2,
                                    replay_buffer_max_length=100, num_eval_episodes=1, eval_interval=2,
                                    log_interval=2, num_races_postproc=1), patch('sys.stdout', new=StringIO()):
                    entry.main()
                output = Path(directory) / 'machine_learning_rl_training/output'
                vse = VSE_REINFORCEMENT(
                    str(output / 'preprocessor_reinforcement_Spa_2019.pkl'),
                    str(output / 'nn_reinforcement_Spa_2019.pt'))
                vse.X_conv = np.zeros((1, 4 + vse.cat_preprocessor.no_transf_cols), dtype=np.float32)
                decisions = vse.make_decision([True], ['A3', 'A4', 'A5'], [True], ['A3'], 0.5)
                self.assertEqual(len(decisions), 1)
                self.assertTrue((output / 'training_curves_Spa_2019.png').is_file())
                self.assertTrue((Path(directory) / 'postproc_results.png').is_file())
            finally:
                os.chdir(previous_directory)

    def test_basic_optimizer_with_updated_cvxpy(self):
        from racesim_basic.src.opt_strategy_basic import opt_strategy_basic
        result = opt_strategy_basic(
            10, {'A': {'k_1_lin': 0.1, 'k_0': 0.0}, 'B': {'k_1_lin': 0.2, 'k_0': 0.0}},
            [['A', 0], ['B', 0]])
        self.assertIsNotNone(result)
        self.assertEqual(int(result.sum()), 10)
        self.assertTrue((result >= 1).all())

    def test_checkpoint_round_trip(self):
        network = QNetwork(5, 4).eval()
        observations = np.random.default_rng(0).random((2, 5), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.pt'
            save_q_network(network, path)
            interpreter = ModelInterpreter(path)
            interpreter.allocate_tensors()
            self.assertEqual(interpreter.get_output_details()[0]['shape'][1], 4)
            interpreter.set_tensor(0, observations)
            interpreter.invoke()
            with torch.no_grad():
                expected = network(torch.from_numpy(observations)).numpy()
            np.testing.assert_allclose(interpreter.get_tensor(0), expected)

    def test_terminal_transition_does_not_bootstrap(self):
        network = QNetwork(2, 3, ())
        target = QNetwork(2, 3, ())
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
            for parameter in target.parameters():
                parameter.fill_(100)
        replay = ReplayBuffer(2, 2)
        replay.add(np.zeros(2), 0, 2.0, 0.0, np.ones(2))
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        self.assertAlmostEqual(update_network(network, target, optimizer, replay, 1), 4.0)

    def test_bundled_tflite_models(self):
        folder = Path(__file__).resolve().parents[1] / 'racesim/input/vse'
        for name in ('nn_supervised_tirechange.tflite', 'nn_supervised_compoundchoice.tflite',
                     'nn_reinforcement_Spa_2019.tflite'):
            with self.subTest(model=name):
                interpreter = ModelInterpreter(folder / name)
                interpreter.allocate_tensors()
                details = interpreter.get_input_details()[0]
                interpreter.set_tensor(details['index'], np.zeros(details['shape'], dtype=np.float32))
                interpreter.invoke()
                result = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
                self.assertTrue(np.isfinite(result).all())

    def test_complete_races_and_short_training(self):
        for multi in (False, True):
            with self.subTest(multi=multi):
                cls = MultiRaceSimulation if multi else RaceSimulation
                kwargs = dict(race_pars_file='pars_Spa_2019.ini', mcs_pars_file='pars_mcs.ini',
                              use_prob_infl=False, create_rand_events=False)
                if not multi:
                    kwargs.update(vse_type='supervised', controlled_driver='HAM')
                env = cls(**kwargs)
                step = env.reset()
                self.assertEqual(step.observation.shape[-1], env.observation_spec().shape[0])
                for _ in range(100):
                    actions = np.zeros(env.race.no_drivers, dtype=int) if multi else 0
                    step = env.step(actions)
                    self.assertTrue(np.isfinite(step.observation).all())
                    self.assertTrue(np.isfinite(step.reward).all())
                    if np.any(step.is_last()):
                        break
                self.assertTrue(np.all(step.is_last()))
                self.assertEqual(env.race.result_status, 0)
                self.assertFalse(np.any(env.step(actions).is_last()))
                network, _, returns, losses = train(
                    env, cls(**kwargs), num_iterations=3, replay_buffer_max_length=100,
                    initial_collect_steps=2, collect_steps_per_iteration=1, fc_layer_params=(8, 8),
                    batch_size=2, learning_rate=1e-3, gamma=1.0, n_step_update=2,
                    target_update_period=1, eval_interval=3, num_eval_episodes=1, log_interval=3)
                self.assertEqual(len(losses), 3)
                self.assertTrue(np.isfinite(losses).all())
                self.assertTrue(np.isfinite(returns).all())
                self.assertEqual(network.no_actions, int(env.action_spec().maximum) + 1)


if __name__ == '__main__':
    unittest.main()
