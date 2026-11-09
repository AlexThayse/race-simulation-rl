"""Q-network architecture and portable PyTorch state-dict checkpoints."""

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, observation_size, no_actions, hidden_sizes=(64, 64)):
        super().__init__()
        self.no_actions = no_actions
        self.observation_size = observation_size
        self.hidden_sizes = tuple(hidden_sizes)
        layers = []
        size = observation_size
        for hidden in hidden_sizes:
            layer = nn.Linear(size, hidden)
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
            layers.extend([layer, nn.ReLU()])
            size = hidden
        output = nn.Linear(size, no_actions)
        nn.init.uniform_(output.weight, -0.03, 0.03)
        nn.init.constant_(output.bias, -0.2)
        layers.append(output)
        self.layers = nn.Sequential(*layers)

    def forward(self, observations):
        return self.layers(observations)


def save_q_network(network, path):
    torch.save(dict(observation_size=network.observation_size, no_actions=network.no_actions,
                    hidden_sizes=network.hidden_sizes, state_dict=network.state_dict()), path)


def load_q_network(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    network = QNetwork(checkpoint['observation_size'], checkpoint['no_actions'], checkpoint['hidden_sizes'])
    network.load_state_dict(checkpoint['state_dict'])
    return network.eval()
