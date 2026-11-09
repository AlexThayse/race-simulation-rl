"""PyTorch DQN with the original Boltzmann exploration and shared multi-driver policy."""

from collections import deque
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from machine_learning.src.q_network import QNetwork


class ReplayBuffer:
    def __init__(self, capacity, observation_size):
        self.observations = np.empty((capacity, observation_size), dtype=np.float32)
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.discounts = np.empty(capacity, dtype=np.float32)
        self.capacity = capacity
        self.size = 0
        self.index = 0

    def add(self, observation, action, reward, discount, next_observation):
        i = self.index
        self.observations[i] = observation
        self.next_observations[i] = next_observation
        self.actions[i] = action
        self.rewards[i] = reward
        self.discounts[i] = discount
        self.index = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(self.size, size=batch_size)
        return tuple(torch.from_numpy(array[indices]) for array in (
            self.observations, self.actions, self.rewards, self.discounts, self.next_observations))


def update_network(network, target, optimizer, replay, batch_size):
    observations, actions, rewards, discounts, next_observations = replay.sample(batch_size)
    with torch.no_grad():
        expected = rewards + discounts * target(next_observations).max(dim=1).values
    predicted = network(observations).gather(1, actions[:, None]).squeeze(1)
    loss = nn.functional.mse_loss(predicted, expected)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def choose_actions(network, observations, temperature=None):
    with torch.inference_mode():
        q_values = network(torch.as_tensor(observations, dtype=torch.float32))
        if temperature is None:
            return q_values.argmax(dim=-1).numpy()
        return torch.distributions.Categorical(logits=q_values / temperature).sample().numpy()


def compute_average_return(env, network, num_episodes):
    returns = []
    for _ in range(num_episodes):
        step = env.reset()
        episode_return = 0.0
        while not np.any(step.is_last()):
            step = env.step(choose_actions(network, step.observation))
            episode_return += np.mean(step.reward)
        returns.append(episode_return)
    return float(np.mean(returns))


def train(train_env, eval_env, *, num_iterations, replay_buffer_max_length, initial_collect_steps,
          collect_steps_per_iteration, fc_layer_params, batch_size, learning_rate, gamma,
          n_step_update, target_update_period, eval_interval, num_eval_episodes, log_interval):
    if n_step_update < 1 or target_update_period < 1:
        raise ValueError('n_step_update and target_update_period must be positive.')
    step = train_env.reset()
    observation_size = step.observation.shape[-1]
    no_actions = int(train_env.action_spec().maximum) + 1
    network = QNetwork(observation_size, no_actions, fc_layer_params)
    target = deepcopy(network).eval()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, eps=1e-7)
    no_drivers = 1 if step.observation.ndim == 1 else step.observation.shape[0]
    replay = ReplayBuffer(replay_buffer_max_length * no_drivers, observation_size)
    pending = [deque() for _ in range(no_drivers)]

    def collect(random_actions=False, temperature=1.0):
        nonlocal step
        if np.any(step.is_last()):
            step = train_env.reset()
        observations = np.atleast_2d(step.observation)
        actions = (np.random.randint(no_actions, size=no_drivers) if random_actions
                   else choose_actions(network, observations, temperature))
        following = train_env.step(int(actions[0]) if no_drivers == 1 else actions)
        next_observations = np.atleast_2d(following.observation)
        rewards = np.atleast_1d(following.reward)
        discounts = np.atleast_1d(following.discount)
        for i, queue in enumerate(pending):
            queue.append((observations[i].copy(), actions[i], rewards[i], discounts[i],
                          next_observations[i].copy()))
            terminal = bool(np.any(following.is_last()))
            while len(queue) >= n_step_update or (terminal and queue):
                reward, discount = 0.0, 1.0
                transitions = list(queue)[:n_step_update]
                for item in transitions:
                    reward += discount * item[2]
                    discount *= gamma * item[3]
                replay.add(queue[0][0], queue[0][1], reward, discount, transitions[-1][4])
                queue.popleft()
        step = following

    for _ in range(max(initial_collect_steps, n_step_update)):
        collect(random_actions=True)
    eval_steps = [0]
    eval_returns = [compute_average_return(eval_env, network, num_eval_episodes)]
    losses = []
    print('INFO: Initial average return: %.3f' % eval_returns[0])
    for iteration in range(1, num_iterations + 1):
        temperature = 1.0 - 0.99 * (iteration - 1) / num_iterations
        for _ in range(collect_steps_per_iteration):
            collect(temperature=temperature)
        loss = update_network(network, target, optimizer, replay, batch_size)
        losses.append(loss)
        if iteration % target_update_period == 0:
            target.load_state_dict(network.state_dict())
        if iteration % log_interval == 0:
            print('INFO: Step %i, loss %.3f' % (iteration, loss))
        if iteration % eval_interval == 0:
            avg_return = compute_average_return(eval_env, network, num_eval_episodes)
            eval_steps.append(iteration)
            eval_returns.append(avg_return)
            print('INFO: Step %i, average return %.3f' % (iteration, avg_return))
    return network.eval(), eval_steps, eval_returns, losses
