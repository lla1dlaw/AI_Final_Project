# Filename: network.py
# Author: Liam Laidlaw
# Created: 11-25-2024
# Description: 


from torch import nn

class Network(nn.module):

    def __init__(self):
        super().__init__()
        # only linear layers
        self.linear = []