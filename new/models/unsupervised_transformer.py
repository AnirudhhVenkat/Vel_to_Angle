import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class UnsupervisedTransformerModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.1, nhead=8, num_layers=4):
        super().__init__()
        
        self.model_type = 'UnsupervisedTransformer'
        self.input_size = input_size
        self.hidden_size = hidden_sizes[-1]
        
        # Ensure hidden size is divisible by number of heads
        assert self.hidden_size % nhead == 0, f"Hidden size ({self.hidden_size}) must be divisible by number of heads ({nhead})"
        
        # Initial normalization of input
        self.input_norm = nn.LayerNorm(input_size)
        
        # Input projection with activation and dropout
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_size, max_len=5000)
        
        # Transformer encoder with gradient checkpointing
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=nhead,
            dim_feedforward=self.hidden_size * 2,  # Reduced from 4x to 2x
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-normalization for better stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_layers,
            norm=nn.LayerNorm(self.hidden_size)
        )
        
        # Progressive output projection
        projection_layers = []
        prev_size = self.hidden_size
        for hidden_size in reversed(hidden_sizes[:-1]):
            projection_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Final output layer
        projection_layers.append(nn.Linear(prev_size, output_size))
        self.output_projection = nn.Sequential(*projection_layers)
        
        # Initialize parameters with smaller values
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
    
    def forward(self, src):
        # Input shape: (batch, seq_len, input_size)
        
        # Normalize input
        x = self.input_norm(src)
        
        # Project input to transformer dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create padding mask for any zero vectors in input
        padding_mask = torch.all(src == 0, dim=-1)
        
        # Apply transformer encoder with padding mask
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Take mean of sequence for more stable training
        x = x.mean(dim=1)
        
        # Project to output dimension
        output = self.output_projection(x)
        
        return output

def pretrain_transformer(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.train()
    total_loss = 0
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            # Handle both tuple and tensor inputs
            if isinstance(batch, (tuple, list)):
                inputs = batch[0]
            else:
                inputs = batch
                
            inputs = inputs.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, inputs)  # Unsupervised learning: reconstruct input
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_epoch_loss = epoch_loss / len(dataloader)
        total_loss = avg_epoch_loss  # Keep track of final loss
        print(f"Pretraining Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
    
    return total_loss

def finetune_transformer(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=10):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        print(f"Finetuning Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict()
    
    model.load_state_dict(best_model_state)
    return model