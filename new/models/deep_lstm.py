import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepLSTM(nn.Module):
    """Enhanced Deep LSTM model with residual connections and layer normalization."""
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.1, bidirectional=True):
        super(DeepLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = len(hidden_sizes)
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        # Input projection with layer normalization
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.LayerNorm(hidden_sizes[0]),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # LSTM layers with layer normalization and residual connections
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(self.num_layers):
            input_dim = hidden_sizes[i-1] * (2 if bidirectional else 1) if i > 0 else hidden_sizes[0]
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_sizes[i],
                    batch_first=True,
                    bidirectional=bidirectional,
                    dropout=dropout if i < self.num_layers - 1 else 0
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_sizes[i] * (2 if bidirectional else 1)))
        
        # Progressive output projection
        final_hidden = hidden_sizes[-1] * (2 if bidirectional else 1)
        self.output_projection = nn.Sequential(
            nn.Linear(final_hidden, final_hidden // 2),
            nn.LayerNorm(final_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(final_hidden // 2, output_size)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.xavier_uniform_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        """
        Forward pass with residual connections and layer normalization.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, output_size)
        """
        # Project input
        x = self.input_projection(x)
        
        # Process through LSTM layers with residual connections
        for i, (lstm, norm) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            # Store residual
            residual = x
            
            # LSTM layer
            x, _ = lstm(x)
            
            # Layer normalization
            x = norm(x)
            
            # Residual connection if shapes match
            if i > 0 and residual.shape == x.shape:
                x = x + residual
            
            # Activation and dropout between layers
            if i < self.num_layers - 1:
                x = F.gelu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Project to output size
        x = self.output_projection(x)
        
        return x
    
    def predict(self, x):
        """Make predictions with the model."""
        self.eval()
        with torch.no_grad():
            return self.forward(x) 