#RNN for classifying genres
#Brandon Rose

import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, bidirectional=True, dropout=0.5):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        #num_layers = number of stacked ltsm layers
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        #store configuration for forward passes
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        #initialize hidden and cell states
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)

        #propagate forward 
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)

        # Take the output of the last time step
        out = self.fc(out[:, -1, :])  # Output shape: (batch_size, num_classes)
        return out

