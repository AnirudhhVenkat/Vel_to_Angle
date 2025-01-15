import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.0):
        super(LSTMModel, self).__init__()
        # Define a stack of LSTM layers
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size if i == 0 else hidden_sizes[i - 1],
                    hidden_sizes[i],
                    batch_first=True)
            for i in range(len(hidden_sizes))
        ])
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(len(hidden_sizes))
        ])
        
        # Final fully connected layer
        self.fc = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        # Input shape: (batch, seq_len, input_size)
        for lstm, dropout in zip(self.lstm_layers, self.dropout_layers):
            x, _ = lstm(x)
            # Only get the last time step after the final LSTM layer
            x = dropout(x)
        # Take the last time step here
        x = x[:, -1, :]
        out = self.fc(x)
        return out