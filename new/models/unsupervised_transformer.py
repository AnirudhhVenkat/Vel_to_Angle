import torch
import torch.nn as nn
import math
import copy
import wandb
import numpy as np
from collections import defaultdict
from tqdm import tqdm

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
    def __init__(self, input_size, hidden_size, nhead, num_layers, output_size=None, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size if output_size is not None else input_size
        
        # Input standardization with larger epsilon
        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.ones(input_size))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
        # Layer normalization with larger epsilon for stability
        self.input_norm = nn.LayerNorm(input_size, eps=1e-2)
        
        # Reduce hidden size and ensure it's even for attention heads
        hidden_size = min(hidden_size, 256)
        if hidden_size % 8 != 0:
            hidden_size = (hidden_size // 8) * 8
        
        # Input projection with very conservative initialization
        self.input_projection = nn.Linear(input_size, hidden_size)
        nn.init.xavier_uniform_(self.input_projection.weight, gain=0.1)
        nn.init.zeros_(self.input_projection.bias)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size)
        
        # Transformer encoder with conservative settings
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,  # Fixed number of heads for stability
            dim_feedforward=2*hidden_size,  # Standard ratio
            dropout=0.1,  # Fixed dropout
            batch_first=True,
            norm_first=False  # Changed to False to avoid nested tensor warning
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, min(num_layers, 3))
        
        # Output projection with conservative initialization
        self.output_projection = nn.Linear(hidden_size, self.output_size)
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.1)
        nn.init.zeros_(self.output_projection.bias)
        
        # Initialize epsilon for numerical stability
        self.eps = 1e-2
    
    def forward(self, x):
        # Input checks and standardization
        if self.training:
            with torch.no_grad():
                mean = x.mean(dim=(0, 1))
                var = x.var(dim=(0, 1), unbiased=False) + self.eps
                self.running_mean = self.running_mean * 0.9 + mean * 0.1
                self.running_var = self.running_var * 0.9 + var * 0.1
                self.num_batches_tracked += 1
                x = (x - mean[None, None, :]) / (torch.sqrt(var[None, None, :] + self.eps))
        else:
            x = (x - self.running_mean[None, None, :]) / (torch.sqrt(self.running_var[None, None, :] + self.eps))
        
        # Apply transformations with safety checks
        x = torch.clamp(x, min=-2, max=2)
        x = self.input_norm(x)
        x = torch.clamp(x, min=-2, max=2)
        
        x = self.input_projection(x)
        x = torch.nn.functional.relu(x)
        x = torch.clamp(x, min=-2, max=2)
        
        x = self.pos_encoder(x)
        x = torch.clamp(x, min=-2, max=2)
        
        # Create attention mask (not padding mask)
        mask = None  # Let transformer handle attention internally
        
        x = self.transformer_encoder(x, mask=mask)
        x = torch.clamp(x, min=-2, max=2)
        
        if self.output_size == self.input_size:
            output = self.output_projection(x)
        else:
            x = torch.nn.functional.adaptive_avg_pool1d(x.transpose(1, 2), 1).squeeze(-1)
            output = self.output_projection(x)
        
        output = torch.clamp(output, min=-2, max=2)
        return output

def pretrain_transformer(model, dataloader, criterion, optimizer, device, num_epochs=10, patience=10):
    """Pretrain transformer model in an unsupervised manner by reconstructing input sequences."""
    model.train()
    best_loss = float('inf')
    best_epoch = -1
    best_model_state = model.state_dict()
    train_losses = []
    patience_counter = 0
    
    # Add gradient clipping
    max_grad_norm = 1.0
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience//2, 
        verbose=True, min_lr=1e-6
    )
    
    print(f"Training with early stopping (patience={patience} epochs)")
    print(f"Gradient clipping at {max_grad_norm}")
    
    # Initialize wandb logging group
    if wandb.run is not None:
        wandb.define_metric("pretrain/*", step_metric="pretrain/epoch")
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc='Pretraining')
    
    for epoch in epoch_pbar:
        epoch_loss = 0
        model.train()
        
        # Enable automatic mixed precision
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
        # Process batches without progress bar
        for batch in dataloader:
            # Handle both tuple and tensor inputs
            if isinstance(batch, (tuple, list)):
                inputs = batch[0]
            else:
                inputs = batch
                
            inputs = inputs.to(device)
            optimizer.zero_grad()
            
            # Use automatic mixed precision if available
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            
            epoch_loss += loss.item()
            
            # Free up memory
            del outputs, loss
            if scaler is not None:
                torch.cuda.empty_cache()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_epoch_loss)
        
        # Update learning rate scheduler
        scheduler.step(avg_epoch_loss)
        
        # Track best loss and epoch, save model state
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            status = "(New best)"
            patience_counter = 0
        else:
            patience_counter += 1
            status = f"(No improvement: {patience_counter})"
        
        # Update progress bar
        epoch_pbar.set_postfix({
            'loss': f'{avg_epoch_loss:.4f}',
            'best': f'{best_loss:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
            'status': status
        })
        
        # Log epoch metrics to wandb
        if wandb.run is not None:
            wandb.log({
                'pretrain/epoch': epoch,
                'pretrain/loss': avg_epoch_loss,
                'pretrain/best_loss': best_loss,
                'pretrain/learning_rate': optimizer.param_groups[0]['lr']
            })
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Restore best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nPretraining completed. Best loss: {best_loss:.4f} at epoch {best_epoch+1}")
    
    return best_loss, best_epoch

def finetune_transformer(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100, patience=10):
    """Finetune transformer model on labeled data with early stopping."""
    try:
        model.to(device)
        best_val_loss = float('inf')
        best_model_state = None
        epochs_without_improvement = 0
        
        # Initialize gradient scaler with new API
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
        # Add gradient clipping
        max_grad_norm = 0.5
        
        print(f"\nStarting finetuning:")
        print(f"Number of epochs: {num_epochs}")
        print(f"Early stopping patience: {patience}")
        print(f"Gradient clipping: {max_grad_norm}")
        print(f"Train loader size: {len(train_loader)}")
        print(f"Validation loader size: {len(val_loader)}")
        
        # Create progress bar for epochs
        epoch_pbar = tqdm(range(num_epochs), desc='Finetuning')
        
        for epoch in epoch_pbar:
            # Training phase
            model.train()
            train_loss = 0
            
            for inputs, targets in train_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                
                # Mixed precision training
                if scaler is not None:
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                
                train_loss += loss.item()
                
                # Clean up memory
                del outputs, loss
                torch.cuda.empty_cache()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0
            
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                for inputs, targets in val_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    del outputs, loss
                    torch.cuda.empty_cache()
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                status = "✓"
            else:
                epochs_without_improvement += 1
                status = f"✗ ({epochs_without_improvement})"
            
            # Update progress bar
            epoch_pbar.set_postfix({
                'train': f'{avg_train_loss:.4f}',
                'val': f'{avg_val_loss:.4f}',
                'best': f'{best_val_loss:.4f}',
                'status': status
            })
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb.log({
                    'finetune/epoch': epoch,
                    'finetune/train_loss': avg_train_loss,
                    'finetune/val_loss': avg_val_loss,
                    'finetune/best_val_loss': best_val_loss
                })
            
            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"\nRestored best model with validation loss: {best_val_loss:.4f}")
        
        return best_val_loss, epoch, model
        
    except Exception as e:
        print(f"\nError in finetune_transformer: {str(e)}")
        import traceback
        traceback.print_exc()
        return float('inf'), -1, None

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