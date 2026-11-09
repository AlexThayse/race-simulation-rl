"""NumPy environment interface shared by single- and multi-driver DQN training."""

from dataclasses import dataclass
import numpy as np


@dataclass
class ArraySpec:
    shape: tuple
    dtype: object
    minimum: float
    maximum: float
    name: str


@dataclass
class TimeStep:
    step_type: object
    reward: object
    discount: object
    observation: np.ndarray

    def is_last(self):
        return np.asarray(self.step_type) == 2


def restart(observation):
    return TimeStep(0, np.float32(0), np.float32(1), observation)


def transition(observation, reward, discount=1.0):
    return TimeStep(1, np.asarray(reward, dtype=np.float32), np.float32(discount), observation)


def termination(observation, reward):
    return TimeStep(2, np.asarray(reward, dtype=np.float32), np.float32(0), observation)


def stack_time_steps(nested_arrays):
    return TimeStep(**{name: np.stack([getattr(step, name) for step in nested_arrays])
                       for name in TimeStep.__dataclass_fields__})


class RaceEnvironment:
    """Keep simulator observations and episode boundaries independent of the ML framework."""

    def reset(self):
        return self._reset()

    def step(self, action):
        return self._step(action)
