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
    """Transformer model with pretraining and finetuning capabilities."""
    def __init__(self, input_size=18, hidden_size=256, nhead=8, num_layers=4, dropout=0.1):
        super(UnsupervisedTransformerModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.pretraining = True
        
        # Calculate embedding sizes
        self.position_dim = 6  # TaG y-positions (one for each of 6 TaGs)
        self.velocity_dim = 12  # 9 moving averages + 3 raw velocities
        
        # Verify dimensions
        if input_size != self.position_dim + self.velocity_dim:
            raise ValueError(f"Input size must be {self.position_dim + self.velocity_dim} (6 TaG y-positions + 12 velocity features)")
        
        print(f"Model initialized with:")
        print(f"- Total input size: {input_size}")
        print(f"- Position features (6): L-F-TaG_y, R-F-TaG_y, L-M-TaG_y, R-M-TaG_y, L-H-TaG_y, R-H-TaG_y")
        print(f"- Velocity features (12):")
        print("  * Moving averages (9): x/y/z_vel_ma5, x/y/z_vel_ma10, x/y/z_vel_ma20")
        print("  * Raw velocities (3): x_vel, y_vel, z_vel")
        print("- Output joint angles during finetuning (48 total):")
        print("  * Front legs (L1, R1): A_flex, A_rot, A_abduct, B_flex, B_rot, C_flex, C_rot, D_flex")
        print("  * Middle legs (L2, R2): A_flex, A_rot, A_abduct, B_flex, B_rot, C_flex, C_rot, D_flex")
        print("  * Hind legs (L3, R3): A_flex, A_rot, A_abduct, B_flex, B_rot, C_flex, C_rot, D_flex")
        
        # Separate embeddings for positions and velocities
        self.position_embedding = nn.Linear(self.position_dim, hidden_size // 2)
        self.velocity_embedding = nn.Linear(self.velocity_dim, hidden_size // 2)
        
        # Combine embeddings
        self.combine_embeddings = nn.Linear(hidden_size, hidden_size)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layers
        self.pretraining_decoder = nn.Linear(hidden_size, input_size)  # Reconstruct all 18 input features
        self.finetuning_decoder = nn.Linear(hidden_size, 48)  # Predict all joint angles (8 per leg × 6 legs)
        
    def forward(self, x):
        # Split input into positions and velocities
        positions = x[..., :self.position_dim]  # First 6 features are TaG y-positions
        velocities = x[..., self.position_dim:]  # Remaining 12 features are velocities
        
        # Embed positions and velocities separately
        pos_embedded = self.position_embedding(positions)
        vel_embedded = self.velocity_embedding(velocities)
        
        # Combine embeddings
        combined = torch.cat([pos_embedded, vel_embedded], dim=-1)
        x = self.combine_embeddings(combined)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transform
        x = self.transformer_encoder(x)
        
        # Decode based on mode
        if self.pretraining:
            output = self.pretraining_decoder(x)  # Reconstruct all 18 input features
        else:
            output = self.finetuning_decoder(x)  # Predict all 48 joint angles
        
        return output
    
    def set_pretraining(self, mode=True):
        """Set the model to pretraining or finetuning mode."""
        self.pretraining = mode

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

def pretrain_transformer(model, train_loader, val_loader, device, config, num_epochs=100, patience=10, trial_dir=None, plots_dir=None, run_name=None):
    """Pretrain transformer model in an unsupervised manner."""
    try:
        # Ensure model is in pretraining mode
        model.set_pretraining(True)
        model.train()
        
        # Initialize loss function and optimizer with reduced learning rate
        criterion = nn.L1Loss().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['pretrain_lr'] * 0.1,  # Reduce learning rate
            weight_decay=config['weight_decay'],
            eps=1e-8  # Increase epsilon for better numerical stability
        )
        
        # Use a more conservative scheduler
        scheduler = torch.optim.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,  # More conservative reduction
            patience=3,
            verbose=True,
            min_lr=1e-6
        )
        
        best_loss = float('inf')
        best_epoch = -1
        best_state = None
        patience_counter = 0
        
        # Enable gradient scaler for mixed precision training
        use_amp = torch.cuda.is_available() and hasattr(torch.cuda.amp, 'autocast')
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # Create progress bar for epochs
        epoch_pbar = tqdm(range(num_epochs), desc='Pretraining')
        
        for epoch in epoch_pbar:
            model.train()
            train_losses = []
            train_metrics = defaultdict(list)
            max_grad_norm = 0.0  # Track maximum gradient norm
            
            # Create progress bar for batches
            batch_pbar = tqdm(train_loader, leave=False, desc=f'Epoch {epoch+1}')
            
            for batch in batch_pbar:
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
                    
                    # Check for NaN loss
                    if torch.isnan(loss):
                        print(f"\nNaN loss detected! Inputs min/max: {inputs.min():.3f}/{inputs.max():.3f}")
                        print(f"Outputs min/max: {outputs.min():.3f}/{outputs.max():.3f}")
                        continue
                
                if use_amp:
                    scaler.scale(loss).backward()
                    
                    # Unscale before gradient clipping
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Skip step if gradients are NaN/inf
                    if torch.isfinite(grad_norm):
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        print(f"\nSkipping step due to infinite gradient norm at epoch {epoch+1}")
                else:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Skip step if gradients are NaN/inf
                    if torch.isfinite(grad_norm):
                        optimizer.step()
                    else:
                        print(f"\nSkipping step due to infinite gradient norm at epoch {epoch+1}")
                
                # Track maximum gradient norm
                if torch.isfinite(grad_norm):
                    max_grad_norm = max(max_grad_norm, grad_norm.item())
                
                train_losses.append(loss.item())
                train_metrics['loss'].append(loss.item())
                
                # Update batch progress bar
                batch_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{np.mean(train_losses):.4f}',
                    'grad_norm': f'{grad_norm.item():.4f}'
                })
                
                # Free up memory
                del outputs, loss
                if use_amp:
                    torch.cuda.empty_cache()
            
            avg_train_loss = sum(train_losses) / len(train_losses)
            train_metrics['avg_loss'] = avg_train_loss
            
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
                        
                        # Skip NaN losses
                        if torch.isfinite(loss):
                            val_loss += loss.item()
                            num_val_batches += 1
                        
                        del outputs, loss
                        if use_amp:
                            torch.cuda.empty_cache()
            
            avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else float('inf')
            
            # Update learning rate scheduler based on validation loss
            scheduler.step(avg_val_loss)
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.4f}',
                'val_loss': f'{avg_val_loss:.4f}',
                'best': f'{best_loss:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                'grad_norm': f'{max_grad_norm:.4f}'
            })
            
            # Log metrics
            if wandb.run is not None:
                wandb.log({
                    'pretrain/train_loss': avg_train_loss,
                    'pretrain/val_loss': avg_val_loss,
                    'pretrain/learning_rate': optimizer.param_groups[0]['lr'],
                    'pretrain/max_grad_norm': max_grad_norm,
                    'pretrain/epoch': epoch
                })
            
            # Check for improvement
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_epoch = epoch
                best_state = {k: v.cpu().clone().detach() for k, v in model.state_dict().items()}
                patience_counter = 0
                print(f"\nNew best model found at epoch {epoch+1} with validation loss: {best_loss:.4f}")
            else:
                patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
        
        # After training loop, ensure we have a model state to return
        if best_state is None:
            print("\nWarning: No best model state was saved during training")
            best_state = {k: v.cpu().clone().detach() for k, v in model.state_dict().items()}
        else:
            print(f"\nPretraining completed. Best loss: {best_loss:.4f} at epoch {best_epoch+1}")
        
        return best_loss, best_epoch, best_state
        
    except Exception as e:
        print(f"Error during pretraining: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        # Ensure we still return something valid even if training fails
        current_state = {k: v.cpu().clone().detach() for k, v in model.state_dict().items()}
        return float('inf'), -1, current_state

def finetune_transformer(model, train_loader, val_loader, device, config, num_epochs=150, patience=10, trial_dir=None, plots_dir=None, run_name=None):
    """Finetune transformer model for joint angle prediction."""
    try:
        # Ensure model is in finetuning mode
        model.set_pretraining(False)
        model.train()
        
        # Initialize loss function and optimizer
        criterion = nn.L1Loss().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['finetune_lr'],  # Use learning rate from config
            weight_decay=config['weight_decay']  # Use weight decay from config
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        best_loss = float('inf')
        best_epoch = -1
        best_state = None
        patience_counter = 0
        
        # Enable gradient scaler for mixed precision training
        use_amp = torch.cuda.is_available() and hasattr(torch.cuda.amp, 'autocast')
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # Create progress bar for epochs
        epoch_pbar = tqdm(range(num_epochs), desc='Finetuning')
        
        for epoch in epoch_pbar:
            model.train()
            train_losses = []
            train_metrics = defaultdict(list)
            
            # Create progress bar for batches
            batch_pbar = tqdm(train_loader, leave=False, desc=f'Epoch {epoch+1}')
            
            for batch in batch_pbar:
                # Handle both tuple and tensor inputs
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch
                else:
                    inputs = batch
                    targets = batch  # For unsupervised case
                    
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                
                # Use AMP context manager if available
                with torch.cuda.amp.autocast() if use_amp else nullcontext():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                
                train_losses.append(loss.item())
                train_metrics['loss'].append(loss.item())
                
                # Update batch progress bar
                batch_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{np.mean(train_losses):.4f}'
                })
                
                # Free up memory
                del outputs, loss
                if use_amp:
                    torch.cuda.empty_cache()
            
            avg_train_loss = sum(train_losses) / len(train_losses)
            train_metrics['avg_loss'] = avg_train_loss
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            num_val_batches = 0
            
            with torch.no_grad():
                with torch.cuda.amp.autocast() if use_amp else nullcontext():
                    for batch in val_loader:
                        # Handle both tuple and tensor inputs
                        if isinstance(batch, (tuple, list)):
                            inputs, targets = batch
                        else:
                            inputs = batch
                            targets = batch  # For unsupervised case
                            
                        inputs = inputs.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                        num_val_batches += 1
                        
                        del outputs, loss
                        if use_amp:
                            torch.cuda.empty_cache()
            
            avg_val_loss = val_loss / num_val_batches
            
            # Update learning rate scheduler based on validation loss
            scheduler.step(avg_val_loss)
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.4f}',
                'val_loss': f'{avg_val_loss:.4f}',
                'best': f'{best_loss:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                'no_improve': patience_counter
            })
            
            # Log metrics
            if wandb.run is not None:
                wandb.log({
                    'finetune/train_loss': avg_train_loss,
                    'finetune/val_loss': avg_val_loss,
                    'finetune/learning_rate': optimizer.param_groups[0]['lr'],
                    'finetune/epoch': epoch
                })
            
            # Check for improvement
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_epoch = epoch
                best_state = {k: v.cpu().clone().detach() for k, v in model.state_dict().items()}
                patience_counter = 0
                print(f"\nNew best model found at epoch {epoch+1} with validation loss: {best_loss:.4f}")
            else:
                patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
        
        # After training loop, ensure we have a model state to return
        if best_state is None:
            print("\nWarning: No best model state was saved during training")
            best_state = {k: v.cpu().clone().detach() for k, v in model.state_dict().items()}
        else:
            print(f"\nFinetuning completed. Best loss: {best_loss:.4f} at epoch {best_epoch+1}")
        
        return best_loss, best_epoch, best_state
        
    except Exception as e:
        print(f"Error during finetuning: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        # Ensure we still return something valid even if training fails
        current_state = {k: v.cpu().clone().detach() for k, v in model.state_dict().items()}
        return float('inf'), -1, current_state

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