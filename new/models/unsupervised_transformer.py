import torch
import torch.nn as nn
import math
import torch.utils.checkpoint
import copy
import wandb
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch.amp  # Add explicit import for amp
import traceback  # Add traceback import
from contextlib import nullcontext  # Add nullcontext import

if __name__ == '__main__':
    # Set multiprocessing start method
    torch.multiprocessing.set_start_method('spawn', force=True)

def check_gpu():
    """Check if GPU is available and print device information."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU")
    return device

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, 1, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # Need shape: [seq_len, batch_size, d_model]
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0)]
        # Return to: [batch_size, seq_len, d_model]
        return self.dropout(x.transpose(0, 1))

class UnsupervisedTransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size=512, nhead=8, num_layers=4, dropout=0.1):
        super(UnsupervisedTransformerModel, self).__init__()
        
        # Input embedding
        self.input_embedding = nn.Linear(input_size, hidden_size)
        
        # Positional encoding is learned
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, hidden_size))  # Max sequence length of 1000
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layers for pretraining (reconstruct input) and finetuning (predict angles)
        self.pretraining_output = nn.Linear(hidden_size, input_size)  # Reconstruct input features
        self.finetuning_output = nn.Linear(hidden_size, 18)  # Predict 18 flexion angles
        
        # Training mode flag
        self.pretraining = True
        
    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        Returns:
            Output tensor of shape (batch_size, seq_len, output_size)
            where output_size is input_size during pretraining or 18 during finetuning
        """
        # Input embedding
        x = self.input_embedding(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Output layer based on mode
        if self.pretraining:
            return self.pretraining_output(x)  # Reconstruct input
        else:
            return self.finetuning_output(x)  # Predict angles
    
    def set_pretraining(self, mode=True):
        """Set the model to pretraining or finetuning mode."""
        self.pretraining = mode
        self.train()  # Ensure model is in training mode

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with scaled positional encodings"""
        position = torch.arange(self.pe.size(1)).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.pe.size(2), 2) * (-math.log(10000.0) / self.pe.size(2)))
        self.pe.data[0, :, 0::2] = torch.sin(position * div_term)
        self.pe.data[0, :, 1::2] = torch.cos(position * div_term)
        self.pe.data = self.pe.data / math.sqrt(self.pe.size(-1))
    
    def forward(self, x):
        """Add learnable positional encoding to input"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class EnhancedTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, batch_first=True):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first
        )
        # Add layer scale parameters
        self.layer_scale_1 = nn.Parameter(torch.ones(1) * 0.1)
        self.layer_scale_2 = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, src):
        """Forward pass with pre-norm architecture and scaled residuals"""
        # Pre-norm attention block
        x = src
        attn = self.self_attn(
            self.norm1(x), 
            self.norm1(x), 
            self.norm1(x)
        )[0]
        x = x + self.dropout1(attn) * self.layer_scale_1
        
        # Pre-norm feedforward block
        ff = self.linear2(
            self.dropout(
                self.activation(
                    self.linear1(self.norm2(x))
                )
            )
        )
        x = x + self.dropout2(ff) * self.layer_scale_2
        
        return x

def pretrain_transformer(model, train_loader, val_loader, device, num_epochs=10, patience=10):
    """Pretrain transformer model in an unsupervised manner by reconstructing input sequences."""
    model.train()
    
    # Use L1Loss (MAE) for both pretraining and finetuning
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )
    
    # Add learning rate scheduler with adjusted patience
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience//3,
        verbose=True, min_lr=1e-6
    )
    
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1
    epochs_no_improve = 0
    
    # Add gradient clipping
    max_grad_norm = 0.5  # Reduced from 1.0 for shorter sequences
    
    # Initialize wandb logging group
    if wandb.run is not None:
        wandb.define_metric("pretrain/*", step_metric="pretrain/epoch")
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc='Pretraining')
    
    # Initialize AMP
    use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    try:
        for epoch in epoch_pbar:
            # Training phase
            model.train()
            train_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                # Handle both tuple and tensor inputs
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch
                    
                inputs = inputs.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                
                # Use AMP context manager if available
                with torch.cuda.amp.autocast() if use_amp else nullcontext():
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)
                
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
                
                # Free up memory
                del outputs, loss
                if use_amp:
                    torch.cuda.empty_cache()
            
            avg_train_loss = train_loss / num_batches
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            num_val_batches = 0
            
            with torch.no_grad():
                with torch.cuda.amp.autocast() if use_amp else nullcontext():
                    for batch in val_loader:
                        # Handle both tuple and tensor inputs
                        if isinstance(batch, (tuple, list)):
                            inputs = batch[0]
                        else:
                            inputs = batch
                            
                        inputs = inputs.to(device, non_blocking=True)
                        outputs = model(inputs)
                        loss = criterion(outputs, inputs)
                        val_loss += loss.item()
                        num_val_batches += 1
                        
                        del outputs, loss
                        if use_amp:
                            torch.cuda.empty_cache()
            
            avg_val_loss = val_loss / num_val_batches
            
            # Update learning rate scheduler based on validation loss
            scheduler.step(avg_val_loss)
            
            # Update progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.4f}',
                'val_loss': f'{avg_val_loss:.4f}',
                'best': f'{best_val_loss:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                'no_improve': epochs_no_improve
            })
            
            # Log metrics
            if wandb.run is not None:
                wandb.log({
                    'pretrain/train_loss': avg_train_loss,
                    'pretrain/val_loss': avg_val_loss,
                    'pretrain/learning_rate': optimizer.param_groups[0]['lr'],
                    'pretrain/epoch': epoch
                })
            
            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = {k: v.cpu().clone().detach() for k, v in model.state_dict().items()}
                best_epoch = epoch
                epochs_no_improve = 0
                print(f"\nNew best model found at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")
            else:
                epochs_no_improve += 1
                
                # Early stopping
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
        
        # After training loop, ensure we have a model state to return
        if best_model_state is None:
            print("\nWarning: No best model state was saved during training")
            best_model_state = {k: v.cpu().clone().detach() for k, v in model.state_dict().items()}
        else:
            print(f"\nPretraining completed. Best loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
        
        return best_val_loss, best_epoch, best_model_state
        
    except Exception as e:
        print(f"Error during pretraining: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        # Ensure we still return something valid even if training fails
        current_state = {k: v.cpu().clone().detach() for k, v in model.state_dict().items()}
        return float('inf'), -1, current_state

def finetune_transformer(model, train_loader, val_loader, device, num_epochs=100, patience=10):
    """Finetune transformer model for joint angle prediction."""
    print("\nStarting finetuning...")
    
    # Use L1Loss (MAE) for both pretraining and finetuning
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )
    
    # Initialize learning rate scheduler with adjusted patience
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience//3,
        verbose=True, min_lr=1e-6
    )
    
    # For monitoring only
    mse_criterion = nn.MSELoss()
    
    # Initialize tracking variables
    best_val_mae = float('inf')
    best_model_state = None
    best_epoch = 0
    epochs_no_improve = 0
    
    # Initialize AMP
    use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    try:
        for epoch in tqdm(range(num_epochs), desc="Finetuning"):
            # Training phase
            model.train()
            train_mae = 0.0
            train_mse = 0.0
            num_batches = 0
            
            for inputs, targets in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                
                # Forward pass with AMP
                with torch.cuda.amp.autocast() if use_amp else nullcontext():
                    outputs = model(inputs)
                    mae = criterion(outputs, targets)  # Primary loss (MAE)
                    mse = mse_criterion(outputs, targets)  # For monitoring
                
                if use_amp:
                    scaler.scale(mae).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    mae.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                
                train_mae += mae.item()
                train_mse += mse.item()
                num_batches += 1
                
                # Clean up memory
                del outputs, mae, mse
                if use_amp:
                    torch.cuda.empty_cache()
            
            avg_train_mae = train_mae / num_batches
            avg_train_mse = train_mse / num_batches
            
            # Validation phase
            model.eval()
            val_mae = 0.0
            val_mse = 0.0
            num_val_batches = 0
            
            with torch.no_grad():
                with torch.cuda.amp.autocast() if use_amp else nullcontext():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)
                        outputs = model(inputs)
                        
                        mae = criterion(outputs, targets)
                        mse = mse_criterion(outputs, targets)
                        
                        val_mae += mae.item()
                        val_mse += mse.item()
                        num_val_batches += 1
                        
                        del outputs, mae, mse
                        if use_amp:
                            torch.cuda.empty_cache()
            
            avg_val_mae = val_mae / num_val_batches
            avg_val_mse = val_mse / num_val_batches
            
            # Update learning rate scheduler based on validation MAE
            scheduler.step(avg_val_mae)
            
            # Update progress bar
            tqdm.set_postfix({
                'train_mae': f'{avg_train_mae:.4f}',
                'val_mae': f'{avg_val_mae:.4f}',
                'best': f'{best_val_mae:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                'no_improve': epochs_no_improve
            })
            
            # Log metrics
            if wandb.run is not None:
                wandb.log({
                    'finetune/train_mae': avg_train_mae,
                    'finetune/train_mse': avg_train_mse,
                    'finetune/val_mae': avg_val_mae,
                    'finetune/val_mse': avg_val_mse,
                    'finetune/learning_rate': optimizer.param_groups[0]['lr'],
                    'finetune/epoch': epoch
                })
            
            # Check for improvement
            if avg_val_mae < best_val_mae:
                best_val_mae = avg_val_mae
                best_model_state = {k: v.cpu().clone().detach() for k, v in model.state_dict().items()}
                best_epoch = epoch
                epochs_no_improve = 0
                print(f"\nNew best model found at epoch {epoch+1} with validation MAE: {best_val_mae:.4f}")
            else:
                epochs_no_improve += 1
                
                # Early stopping
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
        
        # After training loop, ensure we have a model state to return
        if best_model_state is None:
            print("\nWarning: No best model state was saved during training")
            best_model_state = {k: v.cpu().clone().detach() for k, v in model.state_dict().items()}
        else:
            model.load_state_dict(best_model_state)
            print(f"\nFinetuning completed. Best MAE: {best_val_mae:.4f} at epoch {best_epoch+1}")
        
        return best_val_mae, best_epoch, model
        
    except Exception as e:
        print(f"Error during finetuning: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        # Ensure we still return something valid even if training fails
        current_state = {k: v.cpu().clone().detach() for k, v in model.state_dict().items()}
        model.load_state_dict(current_state)
        return float('inf'), -1, model

def train_model(model_params, train_loader, val_loader=None, num_epochs=50):
    """Train the transformer model with the given parameters."""
    # Get device
    device = check_gpu()
    
    # Increase batch size if GPU is available
    if torch.cuda.is_available():
        recommended_batch_size = 512  # Increased from 128
        print(f"GPU detected - using larger batch size: {recommended_batch_size}")
    else:
        recommended_batch_size = 128
        print(f"CPU detected - using standard batch size: {recommended_batch_size}")
    
    model_params['batch_size'] = recommended_batch_size