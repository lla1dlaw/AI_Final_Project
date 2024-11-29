# Filename: network.py
# Author: Liam Laidlaw
# Created: 11-25-2024
# Description: Full feedforward linear network with no convolutions or recurrent structures involved


from torch import nn

class Network(nn.module):
    def __init__(self, num_linear_layers: int, hidden_size: int, in_size: int, out_size: int):
        super().__init__()
        # only linear layers
        self.linear = [nn.Linear(in_size, hidden_size)]
        self.linear.extend([nn.Linear(hidden_size, hidden_size) for _ in range(num_linear_layers)])
        self.linear.append(nn.Linear(hidden_size, out_size))
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.linear:
            x = layer(x)
            x = self.activation(x)
        return x
        