# Filename: main.py
# Author: Liam Laidlaw
# Created: 11-25-2024
# Description: The main driver program for the classifier

import torch
import torchaudio.datasets
import torchaudio
from network import Network


def main():
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    print(f"Using {device} device")

    num_linear_layers = 0
    linear_width = 0
    in_size = 0
    out_size = 0 

    print("Creating linear network")
    linear_net = Network(num_linear_layers, linear_width, in_size, out_size)

    




if __name__ == '__main__':
    main()