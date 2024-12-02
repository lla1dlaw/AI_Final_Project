#RNN for classifying genres
#Brandon Rose

import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Define an LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Ensure x is batched: (batch_size, sequence_length, input_size)
        if x.ndim == 2:  # If x is unbatched
            x = x.unsqueeze(0)

        # Initialize hidden and cell states
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Take the output of the last time step
        out = self.fc(out[:, -1, :])  # Output shape: (batch_size, num_classes)
        return out


