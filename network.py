# Filename: network.py
# Author: Liam Laidlaw
# Created: 11-04-2024
# Description: Class defining a pytorch network

from torch import nn

class Network(nn.Module):

    def __init__(self, layers: int, hidden_size: int, in_size: int, out_size: int):
        super().__init__()
        self.linear = [nn.Linear(in_size, hidden_size)]
        self.linear.extend([nn.Linear(hidden_size, hidden_size) for i in range(layers)])
        self.linear.append(nn.Linear(hidden_size, out_size))
        self.linear = nn.ModuleList(self.linear)
        self.activation = nn.ReLU()

        self.hidden_size = hidden_size
        self.num_layers = layers
        self.accuracy = 0

    def forward(self, x):
        for layer in self.linear:
            x = layer(x)
            x = self.activation(x)
        return x
        

# made by mario
class AltNetwork(nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.linear = [nn.Linear(in_size, 30)]
        self.linear = [nn.Linear(30, 3)]
        self.linear += [nn.Linear(3, 9)]
        self.linear += [nn.Linear(9, out_size)]
        self.linear = nn.ModuleList(self.linear)
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.linear:
            x = layer(x)
            x = self.activation(x)
        return x
