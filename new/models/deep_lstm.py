import torch.nn as nn

class DeepLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout):
        super(DeepLSTM, self).__init__()
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(input_size if i == 0 else hidden_sizes[i-1]*2,
                    hidden_sizes[i],
                    num_layers=2,
                    batch_first=True,
                    bidirectional=True)
            for i in range(len(hidden_sizes))
        ])
        
        # Batch normalization layers
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_sizes[i] * 2)
            for i in range(len(hidden_sizes))
        ])
        
        # Dropout layers
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(dropout)
            for i in range(len(hidden_sizes))
        ])
        
        # Residual projection layers to match dimensions
        self.residual_projections = nn.ModuleList([
            nn.Linear(input_size if i == 0 else hidden_sizes[i-1] * 2, hidden_sizes[i] * 2)
            for i in range(len(hidden_sizes))
        ])
        
        # Layer normalization layers
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_sizes[i] * 2)
            for i in range(len(hidden_sizes))
        ])
        
        # Fully connected layers for final prediction
        self.fc1 = nn.Linear(hidden_sizes[-1] * 2, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout_fc2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, 64)
        self.bn_fc3 = nn.BatchNorm1d(64)
        self.dropout_fc3 = nn.Dropout(dropout)
        self.output = nn.Linear(64, output_size)

    def forward(self, x):
        # Input shape: (batch, seq_len, input_size)
        batch_size = x.size(0)
        prev_x = x
        
        for i, (lstm, bn, dropout, residual_proj, layer_norm) in enumerate(
            zip(self.lstm_layers, self.bn_layers, self.dropout_layers, 
                self.residual_projections, self.layer_norms)):
            
            # LSTM processing
            lstm_out, _ = lstm(x)
            
            # Take last time step
            last_hidden = lstm_out[:, -1, :]
            
            # Apply batch normalization
            normed = bn(last_hidden)
            
            # Create residual connection
            residual = residual_proj(prev_x[:, -1, :])
            
            # Add residual and apply layer normalization
            combined = layer_norm(normed + residual)
            
            # Apply dropout
            x = dropout(combined)
            
            # Prepare for next layer
            x = x.unsqueeze(1)  # Add sequence dimension back
            prev_x = lstm_out   # Store for next residual connection
        
        # Remove sequence dimension
        x = x.squeeze(1)
        
        # Final fully connected layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.dropout_fc2(x)
        x = self.fc3(x)
        x = self.bn_fc3(x)
        x = self.dropout_fc3(x)
        x = self.output(x)
        
        return x