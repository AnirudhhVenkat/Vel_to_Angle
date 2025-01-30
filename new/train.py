import time
from tqdm import tqdm
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.checkpoint
import wandb
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use Agg backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import os
from datetime import datetime, timedelta
from collections import defaultdict
import copy
import traceback
import shutil
import sys
import io
from contextlib import redirect_stdout, nullcontext
import torch.nn.functional as F
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from models import deep_lstm, transformer, tcn
from models.unsupervised_transformer import (
    UnsupervisedTransformerModel,
    pretrain_transformer,
    finetune_transformer
)
from utils.data import prepare_data, calculate_angular_velocities
from utils.metrics import calculate_metrics

class MAELoss(nn.Module):
    """Simple MAE (L1) loss with MSE tracking for metrics."""
    def __init__(self):
        super(MAELoss, self).__init__()
        self.mae_loss = nn.L1Loss()  # MAE loss
        self.mse_loss = nn.MSELoss()  # For metrics only
        self.batch_count = 0
    
    def forward(self, pred, target):
        """
        Calculate MAE loss between prediction and target sequences.
        Args:
            pred: Predicted sequences (batch_size, seq_len, features)
            target: Target sequences (batch_size, seq_len, features)
        Returns:
            MAE loss value
        """
        # Ensure inputs have the same shape
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
        
        # Calculate losses
        mae_loss = self.mae_loss(pred, target)
        mse_loss = self.mse_loss(pred, target)  # For metrics only
        
        # Print progress occasionally
        self.batch_count += 1
        if self.batch_count % 100 == 0:
            print(f"\nProcessed {self.batch_count} batches")
            print(f"Current losses - MAE: {mae_loss.item():.4f}, MSE: {mse_loss.item():.4f}")
        
        # Store losses for logging
        self.last_losses = {
            'mae': mae_loss.item(),
            'mse': mse_loss.item(),
            'total': mae_loss.item()
        }
        
        return mae_loss

# Suppress wandb output and only initialize in main process
os.environ['WANDB_SILENT'] = 'true'
if __name__ == '__main__':
    with redirect_stdout(io.StringIO()):
        if os.environ.get('WANDB_DISABLED', '').lower() != 'true':
            wandb.login()

def init_wandb(config, run_name):
    """Initialize wandb run."""
    if os.environ.get('WANDB_DISABLED', '').lower() != 'true':
        wandb.init(
            project="joint_angle_prediction",
            name=run_name,
            config=config,
            reinit=True
        )
        print(f"Initialized wandb run: {run_name}")

def log_wandb(metrics, step=None, commit=True):
    """Log metrics to wandb."""
    if wandb.run is not None:
        wandb.log(metrics, step=step, commit=commit)

def finish_wandb():
    """Finish wandb run."""
    if wandb.run is not None:
        wandb.finish()

def plot_predictions(predictions, targets, joint_names, model_name, plots_dir=None, epoch=None, phase='train'):
    """Plot predictions vs targets for each joint angle."""
    try:
        num_joints = len(joint_names)
        fig, axes = plt.subplots(num_joints, 1, figsize=(15, 4*num_joints))
        if num_joints == 1:
            axes = [axes]
        
        metrics = {}
        for i, (joint, ax) in enumerate(zip(joint_names, axes)):
            # Get predictions and targets for this joint
            pred = predictions[:, i].cpu().numpy()
            targ = targets[:, i].cpu().numpy()
            
            # Plot
            ax.plot(targ, label='Target', alpha=0.7)
            ax.plot(pred, label='Prediction', alpha=0.7)
            ax.set_title(f'{joint} Prediction vs Target')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Angle (degrees)')
            ax.legend()
            ax.grid(True)
            
            # Calculate metrics for this joint
            mae = np.mean(np.abs(pred - targ))
            mse = np.mean((pred - targ)**2)
            metrics[f'{joint}_mae'] = mae
            metrics[f'{joint}_mse'] = mse
            
            # Add metrics to plot
            ax.text(0.02, 0.98, f'MAE: {mae:.2f}\nMSE: {mse:.2f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot if directory is provided
        if plots_dir is not None:
            plots_dir = Path(plots_dir)
            plots_dir.mkdir(parents=True, exist_ok=True)
            epoch_str = f'_epoch_{epoch}' if epoch is not None else ''
            plot_path = plots_dir / f'{model_name}_{phase}_predictions{epoch_str}.png'
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    f'plots/{phase}_predictions': wandb.Image(str(plot_path)),
                    **{f'metrics/{phase}/{k}': v for k, v in metrics.items()}
                })
        
        plt.close(fig)
        return metrics
        
    except Exception as e:
        print(f"Error in plot_predictions: {str(e)}")
        traceback.print_exc()
        return {}

def get_model_name(model_name, config):
    """Create descriptive model name with hyperparameters"""
    params = config['model_params']
    hidden_sizes = 'x'.join(map(str, params['hidden_sizes']))
    base_name = f"{model_name}_h{hidden_sizes}_d{params['dropout']}"
    
    # Add architecture-specific parameters
    if model_name == 'transformer':
        base_name += f"_head{params['nhead']}_l{params['num_layers']}"
    elif model_name == 'tcn':
        base_name += f"_k{params['kernel_size']}"
    
    return base_name

def objective(trial, model_name, base_config, trial_num, total_trials):
    """Objective function for optimizing model hyperparameters."""
    try:
        # Define hyperparameter search space
        config = base_config.copy()
        config['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        config['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        config['dropout'] = trial.suggest_float('dropout', 0.1, 0.5)
        
        # Define model-specific parameters
        hidden_sizes = []
        for i in range(3):  # 3 hidden layers
            hidden_sizes.append(trial.suggest_categorical(f'hidden_size_{i}', [64, 128, 192, 256, 320, 384, 448, 512]))
        
        # Update model parameters with new input/output dimensions
        config['model_params'] = {
            'input_size': 15,  # 15 input features
            'hidden_sizes': hidden_sizes,
            'output_size': 6,  # 6 joint angles
            'dropout': config['dropout']
        }
        
        # Add model-specific parameters
        if model_name == 'transformer':
            config['model_params'].update({
                'nhead': trial.suggest_categorical('nhead', [4, 8, 16]),
                'num_layers': trial.suggest_int('num_layers', 2, 6)
            })
        
        if model_name == 'tcn':
            config['model_params']['kernel_size'] = trial.suggest_categorical('kernel_size', [3, 5, 7])
        
        print(f"\nTrial {trial_num}/{total_trials}")
        print(f"\n{model_name.upper()} Model Configuration:")
        print(f"Input size: {config['model_params']['input_size']}")
        print(f"Hidden sizes: {config['model_params']['hidden_sizes']}")
        print(f"Output size: {config['model_params']['output_size']}")
        if model_name == 'tcn':
            print(f"Kernel size: {config['model_params']['kernel_size']}")
        elif model_name == 'transformer':
            print(f"Number of heads: {config['model_params']['nhead']}")
            print(f"Number of layers: {config['model_params']['num_layers']}")
        print(f"Dropout: {config['dropout']}")
        
        final_val_loss = train_model(model_name, config)
        
        if final_val_loss is None:
            raise ValueError("Training returned None")
        
        return final_val_loss
    
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')

def check_gpu():
    """Check if GPU is available and print device information."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\nGPU Information:")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Set GPU optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Force some GPU memory allocation to verify it's working
        print("\nTesting GPU Memory Allocation:")
        # Create a significant tensor to force memory allocation
        test_tensor = torch.zeros((1000, 1000, 100), device=device)
        print(f"After allocating test tensor:")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        
        # Clear test tensor
        del test_tensor
        torch.cuda.empty_cache()
        print(f"\nAfter clearing test tensor:")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        
        def print_gpu_memory():
            print(f"\nCurrent GPU Memory Stats:")
            print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
            print(f"Max Allocated: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
        
        return device, print_gpu_memory
    else:
        print("No GPU available, using CPU")
        return torch.device("cpu"), None

def train_model(model_name, config):
    """Train a model with the given configuration."""
    try:
        device, print_gpu_memory = check_gpu()
        config['device'] = device
        
        start_time = time.time()
        model_desc = get_model_name(model_name, config)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{model_desc}_{timestamp}"
        
        # Initialize wandb with complete config
        init_wandb(config, run_name)
        
        # Create trial directory
        models_dir = Path('models')
        models_dir.mkdir(parents=True, exist_ok=True)
        trials_dir = models_dir / 'trials'
        trials_dir.mkdir(parents=True, exist_ok=True)
        trial_dir = trials_dir / run_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        # Train model
        train_loader, val_loader, test_loader = prepare_data(config)
        if not train_loader or not val_loader or not test_loader:
            raise ValueError("Data loaders are empty or not initialized properly.")
        
        model_classes = {
            'deep_lstm': deep_lstm.DeepLSTM,
            'transformer': transformer.TransformerModel,
            'tcn': tcn.TCNModel
        }
        
        model = model_classes[model_name](**config['model_params'])
        model = model.to(device)
        
        if print_gpu_memory:
            print("\nAfter model initialization:")
            print_gpu_memory()
        
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        
        # Use L1Loss (MAE) as criterion
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        initial_epochs = config['initial_epochs']
        patience = config['patience']
        best_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        best_epoch = 0
        
        # Log initial config
        if wandb.run is not None:
            wandb.config.update({
                'model_name': model_name,
                'model_params': config['model_params'],
                'optimizer': optimizer.__class__.__name__,
                'learning_rate': config['learning_rate'],
                'batch_size': config['batch_size'],
                'criterion': 'MAE (L1Loss)',
                'device': str(device)
            })
        
        # Initialize AMP
        use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        global_step = 0
        for epoch in range(config['num_epochs']):
            model.train()
            train_loss = 0.0
            num_train_batches = 0
            
            # Create progress bar for training
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}')
            
            # Training loop
            for inputs, targets in train_pbar:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                
                # Use updated AMP context manager syntax
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
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
                
                train_loss += loss.item()
                num_train_batches += 1
                global_step += 1
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{(train_loss / num_train_batches):.4f}'
                })
                
                # Log batch-level metrics
                if wandb.run is not None:
                    wandb.log({
                        'train/batch_loss': loss.item(),
                        'train/learning_rate': optimizer.param_groups[0]['lr']
                    }, step=global_step)
                
                # Clean up memory
                del outputs, loss
                if use_amp:
                    torch.cuda.empty_cache()
            
            # Calculate average training loss for the epoch
            train_loss /= num_train_batches
            
            # Validation loop
            model.eval()
            val_loss = 0.0
            num_val_batches = 0
            
            # Create progress bar for validation
            val_pbar = tqdm(val_loader, desc='Validation')
            
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                    for inputs, targets in val_pbar:
                        inputs = inputs.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                        num_val_batches += 1
                        
                        # Update validation progress bar
                        val_pbar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'avg_loss': f'{(val_loss / num_val_batches):.4f}'
                        })
                        
                        # Clean up memory
                        del outputs, loss
                        if use_amp:
                            torch.cuda.empty_cache()
            
            val_loss /= num_val_batches
            
            # Log epoch-level metrics
            if wandb.run is not None:
                wandb.log({
                    'train/epoch_loss': train_loss,
                    'val/epoch_loss': val_loss,
                    'epoch': epoch
                }, step=global_step)
            
            print(f'Epoch {epoch+1}/{config["num_epochs"]} - '
                  f'Train Loss: {train_loss:.4f} - '
                  f'Val Loss: {val_loss:.4f}')
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = model.state_dict()
                best_epoch = epoch
                epochs_no_improve = 0
                
                # Log best model metrics
                if wandb.run is not None:
                    wandb.log({
                        'best/val_loss': best_loss,
                        'best/epoch': best_epoch
                    }, step=global_step)
                
                # Save best model
                model_save_path = trial_dir / 'best_model.pth'
                torch.save({
                    'model_state_dict': best_model_state,
                    'config': config,
                    'best_loss': best_loss,
                    'best_epoch': best_epoch,
                    'metrics': {
                        'train_loss': train_loss,
                        'val_loss': best_loss,
                        'best_epoch': best_epoch,
                        'total_epochs': epoch + 1
                    }
                }, model_save_path)
                
                print(f"\nNew best model found at epoch {epoch+1}!")
                print(f"Val Loss: {best_loss:.4f}")
                print(f"Saved best model to: {model_save_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience and epoch >= initial_epochs:
                    print(f'\nEarly stopping triggered after {epoch+1} epochs')
                    break
        
        # Final evaluation and plotting
        model.load_state_dict(best_model_state)
        model.eval()
        
        print("\nGenerating final predictions and plots...")
        test_predictions = []
        test_targets = []
        test_loss = 0.0
        
        # Create progress bar for testing
        test_pbar = tqdm(test_loader, desc='Testing')
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                for inputs, targets in test_pbar:
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    test_loss += loss.item()
                    test_predictions.append(outputs.cpu().numpy())
                    test_targets.append(targets.cpu().numpy())
                    
                    # Update test progress bar
                    test_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{(test_loss / (test_pbar.n + 1)):.4f}'
                    })
                    
                    # Clean up memory
                    del outputs, loss
                    if use_amp:
                        torch.cuda.empty_cache()
        
        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss}')
        
        test_predictions = np.concatenate(test_predictions, axis=0)
        test_targets = np.concatenate(test_targets, axis=0)
        
        # Generate and log prediction plots
        metrics = plot_predictions(
            test_predictions, 
            test_targets, 
            joint_names=config['joint_angle_columns'],
            model_name=model_name,
            plots_dir=None,
            epoch=None,
            phase='test'
        )
        
        # Log final metrics
        log_wandb({
            'test/loss': test_loss,
            'test/metrics': metrics,
            'training/total_epochs': config['num_epochs'],
            'training/best_epoch': best_epoch,
            'training/best_val_loss': best_loss,
        })
        
        # Save final model and results
        final_results = {
            'model_state_dict': best_model_state,
            'config': config,
            'metrics': metrics,
            'best_loss': best_loss,
            'best_epoch': best_epoch,
            'total_epochs': config['num_epochs'],
            'test_loss': test_loss
        }
        
        torch.save(final_results, trial_dir / 'final_model.pth')
        
        # Finish wandb run
        finish_wandb()
        
        return best_loss
    
    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        finish_wandb()
        return None

def get_default_config():
    """Get default configuration for the model."""
    return {
        'data_path': r'Z:\Divya\TEMP_transfers\toAni\BPN_P9LT_P9RT_flyCoords.csv',
        'batch_size': 32,  # Reduced from 64 to prevent memory issues
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'dropout': 0.1,
        'num_epochs': 100,
        'patience': 15,
        'initial_epochs': 10,  # Minimum number of epochs before early stopping
        
        # Data processing options
        'use_windows': True,  # Enable windowed data
        'window_size': 50,    # Size of the sliding window
        
        'model_params': {
            'input_size': 10,  # 10 velocity features
            'hidden_sizes': [128, 128],  # Reduced from 256 to prevent memory issues
            'output_size': 18,  # 18 flexion angles
            'dropout': 0.1,
            'nhead': 8,
            'num_layers': 4
        },
        
        # All 18 flexion angles
        'joint_angle_columns': [
            'L1A_flex', 'L1B_flex', 'L1C_flex',
            'L2A_flex', 'L2B_flex', 'L2C_flex',
            'L3A_flex', 'L3B_flex', 'L3C_flex',
            'R1A_flex', 'R1B_flex', 'R1C_flex',
            'R2A_flex', 'R2B_flex', 'R2C_flex',
            'R3A_flex', 'R3B_flex', 'R3C_flex'
        ],
        
        # Best correlated velocity features
        'velocity_features': [
            # Z-velocity features (strongest correlations)
            'z_vel', 'z_vel_ma5', 'z_vel_ma10', 'z_vel_ma20',
            # X-velocity features (moderate correlations)
            'x_vel', 'x_vel_ma5', 'x_vel_ma10', 'x_vel_ma20',
            # Combined velocities
            'velocity_magnitude', 'xz_velocity'
        ]
    }

def train_unsupervised_transformer(config):
    """Train the unsupervised transformer model with pretraining and finetuning."""
    try:
        # Create descriptive run name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"unsupervised_transformer_{timestamp}"
        
        # Create directories
        models_dir = Path('models')
        models_dir.mkdir(parents=True, exist_ok=True)
        trials_dir = models_dir / 'trials'
        trials_dir.mkdir(parents=True, exist_ok=True)
        trial_dir = trials_dir / run_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        plots_dir = Path('plots')
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb with more descriptive name
        if os.environ.get('WANDB_DISABLED', '').lower() != 'true':
            wandb.init(
                project="joint_angle_prediction",
                name=run_name,
                config=config,
                reinit=True
            )
        
        # Load and prepare data
        train_loader, val_loader, test_loader, feature_scaler, target_scaler = prepare_data(config)
        
        # Get input size and number of joints from the first batch
        sample_batch = next(iter(train_loader))
        input_size = sample_batch[0].shape[-1]  # Features
        num_joints = sample_batch[1].shape[-1]  # Joint angles
        print(f"\nInput size from data: {input_size}")
        print(f"Number of joints: {num_joints}")
        print(f"Number of features: {len(config['velocity_features'])}")
        print("Features used:", config['velocity_features'])
        
        # Initialize model
        model = UnsupervisedTransformerModel(
            input_size=input_size,
            num_joints=num_joints,
            hidden_size=config['model_params'].get('hidden_size', 512),
            nhead=config['model_params'].get('nhead', 8),
            num_layers=config['model_params'].get('num_layers', 4),
            dropout=config['model_params'].get('dropout', 0.1)
        )
        
        if torch.cuda.is_available():
            model = model.cuda()
            print("Using CUDA")
        
        # Training history for plotting
        history = {
            'pretrain_train_loss': [],
            'pretrain_val_loss': [],
            'finetune_train_loss': [],
            'finetune_val_loss': [],
            'pretrain_epochs': [],
            'finetune_epochs': []
        }
        
        # Pretraining phase
        print("\nStarting pretraining phase...")
        model.set_pretraining(True)
        
        pretrain_config = {
            'num_epochs': config.get('pretrain_epochs', 100),
            'learning_rate': config.get('learning_rate', 1e-4),
            'weight_decay': config.get('weight_decay', 0.01),
            'batch_size': config.get('batch_size', 64),
            'patience': config.get('patience', 15)
        }
        
        criterion = MAELoss().to(model.device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=pretrain_config['learning_rate'],
            weight_decay=pretrain_config['weight_decay']
        )
        
        # Enable gradient scaler and AMP for mixed precision
        use_amp = torch.cuda.is_available() and hasattr(torch.cuda.amp, 'autocast')
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        best_pretrain_loss = float('inf')
        best_pretrain_epoch = -1
        best_pretrain_state = None
        patience_counter = 0
        
        for epoch in range(pretrain_config['num_epochs']):
            model.train()
            train_losses = []
            train_metrics = defaultdict(list)
            
            for batch_features, batch_targets in tqdm(train_loader, desc=f'Pretrain Epoch {epoch+1}'):
                batch_features = batch_features.float()
                batch_targets = batch_targets.float()
                
                if torch.cuda.is_available():
                    batch_features = batch_features.cuda()
                    batch_targets = batch_targets.cuda()
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(batch_features)
                    if model.pretraining:
                        loss = criterion(outputs, batch_features)
                    else:
                        loss = criterion(outputs, batch_targets)
                
                if scaler is not None:
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
                for loss_name, loss_value in criterion.last_losses.items():
                    train_metrics[f'train_{loss_name}_loss'].append(loss_value)
                
                # Update progress bar
                train_loop.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mae': f'{criterion.last_losses["mae"]:.4f}',
                    'mse': f'{criterion.last_losses["mse"]:.4f}'
                })
                
                # Clean up memory
                del outputs, loss
                if use_amp:
                    torch.cuda.empty_cache()
            
            # Validation phase
            model.eval()
            val_losses = []
            val_metrics = defaultdict(list)
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.float()
                    batch_targets = batch_targets.float()
                    
                    if torch.cuda.is_available():
                        batch_features = batch_features.cuda()
                        batch_targets = batch_targets.cuda()
                    
                    with torch.cuda.amp.autocast(device_type='cuda', enabled=use_amp):
                        outputs = model(batch_features)
                        loss = criterion(outputs, batch_targets)
                    
                    val_losses.append(loss.item())
                    for loss_name, loss_value in criterion.last_losses.items():
                        val_metrics[f'val_{loss_name}_loss'].append(loss_value)
                    
                    # Update validation progress bar
                    val_loop.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'mae': f'{criterion.last_losses["mae"]:.4f}',
                        'mse': f'{criterion.last_losses["mse"]:.4f}'
                    })
                    
                    # Clean up memory
                    del outputs, loss
                    if use_amp:
                        torch.cuda.empty_cache()
            
            # Calculate average losses
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            # Calculate average metrics
            train_metric_avgs = {name: np.mean(values) for name, values in train_metrics.items()}
            val_metric_avgs = {name: np.mean(values) for name, values in val_metrics.items()}
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Log metrics
            if wandb.run is not None:
                wandb.log({
                    'pretrain/loss/train': avg_train_loss,
                    'pretrain/loss/val': avg_val_loss,
                    'pretrain/learning_rate': optimizer.param_groups[0]['lr'],
                    'pretrain/metrics/train_mae': train_metric_avgs["train_mae_loss"],
                    'pretrain/metrics/train_mse': train_metric_avgs["train_mse_loss"],
                    'pretrain/metrics/val_mae': val_metric_avgs["val_mae_loss"],
                    'pretrain/metrics/val_mse': val_metric_avgs["val_mse_loss"],
                    'pretrain/epoch': epoch
                })
            
            # Print epoch summary
            print(f'\nEpoch {epoch+1}/{pretrain_config["num_epochs"]}:')
            print(f'Train Loss: {avg_train_loss:.4f} (MAE: {train_metric_avgs["train_mae_loss"]:.4f}, '
                  f'MSE: {train_metric_avgs["train_mse_loss"]:.4f})')
            print(f'Val Loss: {avg_val_loss:.4f} (MAE: {val_metric_avgs["val_mae_loss"]:.4f}, '
                  f'MSE: {val_metric_avgs["val_mse_loss"]:.4f})')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save best model
            if avg_val_loss < best_pretrain_loss:
                best_pretrain_loss = avg_val_loss
                best_pretrain_epoch = epoch
                best_pretrain_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
                
                # Save best pretrained model
                pretrain_path = trial_dir / 'best_pretrained_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_pretrain_state,
                    'val_loss': best_pretrain_loss,
                    'config': config
                }, pretrain_path)
                print(f"\nSaved best pretrained model to: {pretrain_path}")
            else:
                patience_counter += 1
                if patience_counter >= pretrain_config['patience']:
                    print(f'Early stopping pretraining after {epoch+1} epochs')
                    break
        
        # Plot pretraining history
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history['pretrain_epochs'], history['pretrain_train_loss'], label='Train Loss')
        ax.plot(history['pretrain_epochs'], history['pretrain_val_loss'], label='Val Loss')
        ax.set_title('Pretraining Loss History')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        
        # Save pretraining plot
        pretrain_plot_path = plots_dir / 'pretrain_history.png'
        pretrain_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pretrain_plot_path, bbox_inches='tight', dpi=300)
        if wandb.run is not None:
            wandb.log({
                'plots/pretrain_loss': wandb.Image(str(pretrain_plot_path)),
                'pretrain/final_train_loss': history['pretrain_train_loss'][-1],
                'pretrain/final_val_loss': history['pretrain_val_loss'][-1],
                'pretrain/best_val_loss': best_pretrain_loss,
                'pretrain/best_epoch': best_pretrain_epoch,
                'pretrain/total_epochs': len(history['pretrain_epochs'])
            })
        plt.close(fig)
        
        # Finetuning phase
        print("\nStarting finetuning phase...")
        model.set_pretraining(False)
        model.load_state_dict(best_pretrain_state)
        
        # New optimizer and scheduler for finetuning
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=pretrain_config['learning_rate'] * 0.1,
            weight_decay=pretrain_config['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        best_finetune_loss = float('inf')
        best_finetune_epoch = -1
        best_finetune_state = None
        patience_counter = 0
        
        for epoch in range(config.get('finetune_epochs', 150)):
            model.train()
            train_losses = []
            train_metrics = defaultdict(list)
            
            for batch_features, batch_targets in tqdm(train_loader, desc=f'Finetune Epoch {epoch+1}'):
                batch_features = batch_features.float()
                batch_targets = batch_targets.float()
                
                if torch.cuda.is_available():
                    batch_features = batch_features.cuda()
                    batch_targets = batch_targets.cuda()
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)
                
                if scaler is not None:
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
                for loss_name, loss_value in criterion.last_losses.items():
                    train_metrics[f'train_{loss_name}_loss'].append(loss_value)
                
                # Update progress bar
                train_loop.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mae': f'{criterion.last_losses["mae"]:.4f}',
                    'mse': f'{criterion.last_losses["mse"]:.4f}'
                })
                
                # Clean up memory
                del outputs, loss
                if use_amp:
                    torch.cuda.empty_cache()
            
            model.eval()
            val_losses = []
            val_metrics = defaultdict(list)
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.float().to(device)
                    batch_targets = batch_targets.float().to(device)
                    
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        outputs = model(batch_features)
                        loss = criterion(outputs, batch_targets)
                    
                    val_losses.append(loss.item())
                    for loss_name, loss_value in criterion.last_losses.items():
                        val_metrics[f'val_{loss_name}_loss'].append(loss_value)
                    
                    # Update validation progress bar
                    val_loop.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'mae': f'{criterion.last_losses["mae"]:.4f}',
                        'mse': f'{criterion.last_losses["mse"]:.4f}'
                    })
                    
                    # Clean up memory
                    del outputs, loss
                    if use_amp:
                        torch.cuda.empty_cache()
            
            # Calculate average losses
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            # Calculate average metrics
            train_metric_avgs = {name: np.mean(values) for name, values in train_metrics.items()}
            val_metric_avgs = {name: np.mean(values) for name, values in val_metrics.items()}
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Log metrics
            if wandb.run is not None:
                wandb.log({
                    'finetune/loss/train': avg_train_loss,
                    'finetune/loss/val': avg_val_loss,
                    'finetune/learning_rate': optimizer.param_groups[0]['lr'],
                    'finetune/metrics/train_mae': train_metric_avgs["train_mae_loss"],
                    'finetune/metrics/train_mse': train_metric_avgs["train_mse_loss"],
                    'finetune/metrics/val_mae': val_metric_avgs["val_mae_loss"],
                    'finetune/metrics/val_mse': val_metric_avgs["val_mse_loss"],
                    'finetune/epoch': epoch
                })
            
            # Print epoch summary
            print(f'\nEpoch {epoch+1}/{config.get("finetune_epochs", 150)}:')
            print(f'Train Loss: {avg_train_loss:.4f} (MAE: {train_metric_avgs["train_mae_loss"]:.4f}, '
                  f'MSE: {train_metric_avgs["train_mse_loss"]:.4f})')
            print(f'Val Loss: {avg_val_loss:.4f} (MAE: {val_metric_avgs["val_mae_loss"]:.4f}, '
                  f'MSE: {val_metric_avgs["val_mse_loss"]:.4f})')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save best model
            if avg_val_loss < best_finetune_loss:
                best_finetune_loss = avg_val_loss
                best_finetune_epoch = epoch
                best_finetune_state = model.state_dict()
                patience_counter = 0
                
                # Save best finetuned model
                finetune_path = trial_dir / 'best_finetuned_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_finetune_state,
                    'pretrained_state_dict': best_pretrain_state,
                    'val_loss': best_finetune_loss,
                    'pretrain_val_loss': best_pretrain_loss,
                    'config': config,
                    'metrics': {
                        'pretrain_best_loss': best_pretrain_loss,
                        'pretrain_best_epoch': best_pretrain_epoch,
                        'finetune_best_loss': best_finetune_loss,
                        'finetune_best_epoch': best_finetune_epoch
                    }
                }, finetune_path)
                print(f"\nSaved best finetuned model to: {finetune_path}")
            else:
                patience_counter += 1
                if patience_counter >= config.get('patience', 15):
                    print(f'\nEarly stopping triggered after {epoch+1} epochs')
                    break
        
            # Generate prediction plots every 10 epochs
            if epoch % 10 == 0 and plots_dir is not None:
                model.eval()
                with torch.no_grad():
                    # Get a batch of validation data for plotting
                    val_features, val_targets = next(iter(val_loader))
                    val_features = val_features.float().to(device)
                    val_targets = val_targets.float().to(device)
                    
                    # Generate predictions
                    predictions = model(val_features)
                    
                    # Plot and log
                    plot_predictions(
                        predictions=predictions,
                        targets=val_targets,
                        joint_names=config['joint_angle_columns'],
                        model_name='transformer',
                        plots_dir=plots_dir,
                        epoch=epoch,
                        phase='val'
                    )
        
        # Generate final prediction plots
        model.eval()
        with torch.no_grad():
            val_features, val_targets = next(iter(val_loader))
            val_features = val_features.float().to(device)
            val_targets = val_targets.float().to(device)
            predictions = model(val_features)
            
            final_metrics = plot_predictions(
                predictions=predictions,
                targets=val_targets,
                joint_names=config['joint_angle_columns'],
                model_name='transformer',
                plots_dir=plots_dir,
                epoch='final',
                phase='val'
            )
            
            if wandb.run is not None:
                wandb.log({
                    'final/metrics': final_metrics
                })
        
        return model, final_metrics
        
    except Exception as e:
        print(f"Error in training: {str(e)}")
        traceback.print_exc()
        if wandb.run is not None:
            wandb.finish()
        raise

def objective_unsupervised(trial, base_config, trial_num, total_trials):
    """Objective function for optimizing unsupervised transformer hyperparameters."""
    try:
        # Get device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create necessary directories
        models_dir = Path('models')
        models_dir.mkdir(parents=True, exist_ok=True)
        trials_dir = models_dir / 'trials'
        trials_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = Path('plots')
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Get default config and update with base_config
        config = base_config.copy()
        
        # Define hyperparameter search space
        config.update({
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'pretrain_lr': trial.suggest_float('pretrain_lr', 1e-4, 1e-3, log=True),
            'finetune_lr': trial.suggest_float('finetune_lr', 1e-5, 1e-4, log=True),
            'dropout': trial.suggest_float('dropout', 0.1, 0.3),
            'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True),
            'window_size': trial.suggest_int('window_size', 20, 200, step=10),
            'hidden_size': trial.suggest_categorical('hidden_size', [256, 512, 768, 1024]),
            'num_layers': trial.suggest_int('num_layers', 2, 6),
            'nhead': trial.suggest_categorical('nhead', [4, 8, 16])
        })
        
        # Create descriptive run name for this trial
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"trial_{trial_num}_{timestamp}"
        trial_dir = trials_dir / run_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trial config
        with open(trial_dir / 'config.txt', 'w') as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
        
        # Initialize wandb for this trial
        init_wandb(config, run_name)
        
        # Load and prepare data
        train_loader, val_loader, test_loader, feature_scaler, target_scaler = prepare_data(config)
        
        # Get input size from the first batch
        sample_batch = next(iter(train_loader))[0]
        input_size = sample_batch.shape[-1]
        print(f"\nInput size from data: {input_size}")
        
        # Initialize model
        model = UnsupervisedTransformerModel(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(device)
        
        print("\nStarting pretraining phase...")
        model.set_pretraining(True)
        
        # Pretraining
        pretrain_loss, pretrain_epoch, pretrain_state = pretrain_transformer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config,
            num_epochs=config.get('pretrain_epochs', 100),
            patience=config.get('patience', 10),
            trial_dir=trial_dir,
            plots_dir=plots_dir,
            run_name=run_name
        )
        
        if pretrain_loss == float('inf'):
            raise ValueError("Pretraining failed")
        
        print("\nStarting finetuning phase...")
        model.load_state_dict(pretrain_state)
        model.set_pretraining(False)
        
        # Finetuning
        finetune_loss, finetune_epoch, model = finetune_transformer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config,
            num_epochs=config.get('finetune_epochs', 150),
            patience=config.get('patience', 10),
            trial_dir=trial_dir,
            plots_dir=plots_dir,
            run_name=run_name
        )
        
        if finetune_loss == float('inf'):
            raise ValueError("Finetuning failed")
        
        # Save final model state
        torch.save({
            'pretrain_state_dict': pretrain_state,
            'finetune_state_dict': model.state_dict(),
            'config': config,
            'pretrain_loss': pretrain_loss,
            'pretrain_epoch': pretrain_epoch,
            'finetune_loss': finetune_loss,
            'finetune_epoch': finetune_epoch
        }, trial_dir / 'final_model.pth')
        
        # Log final results
        if wandb.run is not None:
            wandb.log({
                'final/pretrain_loss': pretrain_loss,
                'final/pretrain_epoch': pretrain_epoch,
                'final/finetune_loss': finetune_loss,
                'final/finetune_epoch': finetune_epoch
            })
            finish_wandb()
        
        return finetune_loss
        
    except Exception as e:
        print(f"Trial failed: {e}")
        traceback.print_exc()
        if wandb.run is not None:
            wandb.finish()
        return float('inf')

def pretrain_transformer(model, train_loader, val_loader, device, config, num_epochs=100, patience=10, trial_dir=None, plots_dir=None, run_name=None):
    """Pretrain transformer model in an unsupervised manner."""
    try:
        # Ensure model is in pretraining mode
        model.set_pretraining(True)
        model.train()
        
        # Initialize loss function and optimizer
        criterion = MAELoss().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['pretrain_lr'],
            weight_decay=config['weight_decay']
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
        
        for epoch in range(num_epochs):
            model.train()
            train_losses = []
            train_metrics = defaultdict(list)
            
            # Training loop with progress bar
            train_loop = tqdm(train_loader, desc=f'Pretrain Epoch {epoch+1}')
            for batch_idx, (batch_features, _) in enumerate(train_loop):
                batch_features = batch_features.float().to(device)
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    # Forward pass - reconstruct input features
                    reconstructed = model(batch_features)
                    # Calculate reconstruction loss
                    loss = criterion(reconstructed, batch_features)
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                
                # Store losses for logging
                train_losses.append(loss.item())
                for loss_name, loss_value in criterion.last_losses.items():
                    train_metrics[f'train_{loss_name}_loss'].append(loss_value)
                
                # Update progress bar
                train_loop.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mae': f'{criterion.last_losses["mae"]:.4f}',
                    'mse': f'{criterion.last_losses["mse"]:.4f}'
                })
                
                # Clean up memory
                del reconstructed, loss
                if use_amp:
                    torch.cuda.empty_cache()
            
            # Validation loop
            model.eval()
            val_losses = []
            val_metrics = defaultdict(list)
            
            with torch.no_grad():
                val_loop = tqdm(val_loader, desc='Validation')
                for batch_features, _ in val_loop:
                    batch_features = batch_features.float().to(device)
                    
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        reconstructed = model(batch_features)
                        loss = criterion(reconstructed, batch_features)
                    
                    val_losses.append(loss.item())
                    for loss_name, loss_value in criterion.last_losses.items():
                        val_metrics[f'val_{loss_name}_loss'].append(loss_value)
                    
                    # Update validation progress bar
                    val_loop.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'mae': f'{criterion.last_losses["mae"]:.4f}',
                        'mse': f'{criterion.last_losses["mse"]:.4f}'
                    })
                    
                    # Clean up memory
                    del reconstructed, loss
                    if use_amp:
                        torch.cuda.empty_cache()
            
            # Calculate average losses
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            # Calculate average metrics
            train_metric_avgs = {name: np.mean(values) for name, values in train_metrics.items()}
            val_metric_avgs = {name: np.mean(values) for name, values in val_metrics.items()}
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Log metrics
            if wandb.run is not None:
                wandb.log({
                    'pretrain/loss/train': avg_train_loss,
                    'pretrain/loss/val': avg_val_loss,
                    'pretrain/learning_rate': optimizer.param_groups[0]['lr'],
                    'pretrain/metrics/train_mae': train_metric_avgs["train_mae_loss"],
                    'pretrain/metrics/train_mse': train_metric_avgs["train_mse_loss"],
                    'pretrain/metrics/val_mae': val_metric_avgs["val_mae_loss"],
                    'pretrain/metrics/val_mse': val_metric_avgs["val_mse_loss"],
                    'pretrain/epoch': epoch
                })
            
            # Print epoch summary
            print(f'\nEpoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {avg_train_loss:.4f} (MAE: {train_metric_avgs["train_mae_loss"]:.4f}, '
                  f'MSE: {train_metric_avgs["train_mse_loss"]:.4f})')
            print(f'Val Loss: {avg_val_loss:.4f} (MAE: {val_metric_avgs["val_mae_loss"]:.4f}, '
                  f'MSE: {val_metric_avgs["val_mse_loss"]:.4f})')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_epoch = epoch
                best_state = {k: v.cpu().clone().detach() for k, v in model.state_dict().items()}
                patience_counter = 0
                
                # Save best model state
                if trial_dir is not None:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': best_state,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': best_loss,
                        'metrics': {
                            'train': train_metric_avgs,
                            'val': val_metric_avgs
                        }
                    }, trial_dir / 'best_pretrain_model.pth')
                    print(f"\nSaved new best model at epoch {epoch+1} with val_loss: {best_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'\nEarly stopping triggered after {epoch+1} epochs')
                    break
            
            # Generate prediction plots every 10 epochs
            if epoch % 10 == 0 and plots_dir is not None:
                model.eval()
                with torch.no_grad():
                    # Get a batch of validation data for plotting
                    val_features, _ = next(iter(val_loader))
                    val_features = val_features.float().to(device)
                    
                    # Generate reconstructions
                    reconstructed = model(val_features)
                    
                    # Plot and log
                    plot_predictions(
                        predictions=reconstructed,
                        targets=val_features,
                        joint_names=config['velocity_features'],
                        model_name='transformer',
                        plots_dir=plots_dir,
                        epoch=epoch,
                        phase='pretrain_val'
                    )
        
        # Generate final reconstruction plots
        model.eval()
        with torch.no_grad():
            val_features, _ = next(iter(val_loader))
            val_features = val_features.float().to(device)
            reconstructed = model(val_features)
            
            final_metrics = plot_predictions(
                predictions=reconstructed,
                targets=val_features,
                joint_names=config['velocity_features'],
                model_name='transformer',
                plots_dir=plots_dir,
                epoch='final',
                phase='pretrain_val'
            )
            
            if wandb.run is not None:
                wandb.log({
                    'final/pretrain_metrics': final_metrics
                })
        
        return best_loss, best_epoch, best_state
        
    except Exception as e:
        print(f"Error during pretraining: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return float('inf'), -1, None

def finetune_transformer(model, train_loader, val_loader, device, config, num_epochs=150, patience=10, trial_dir=None, plots_dir=None, run_name=None):
    """Finetune transformer model for joint angle prediction."""
    try:
        # Ensure model is in finetuning mode
        model.set_pretraining(False)
        model.train()
        
        # Initialize loss function and optimizer
        criterion = MAELoss().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['finetune_lr'],
            weight_decay=config['weight_decay']
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
        
        for epoch in range(num_epochs):
            model.train()
            train_losses = []
            train_metrics = defaultdict(list)
            
            # Training loop with progress bar
            train_loop = tqdm(train_loader, desc=f'Finetune Epoch {epoch+1}')
            for batch_idx, (batch_features, batch_targets) in enumerate(train_loop):
                batch_features = batch_features.float().to(device)
                batch_targets = batch_targets.float().to(device)
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    # Forward pass - predict joint angles
                    predictions = model(batch_features)
                    # Calculate prediction loss
                    loss = criterion(predictions, batch_targets)
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                
                # Store losses for logging
                train_losses.append(loss.item())
                for loss_name, loss_value in criterion.last_losses.items():
                    train_metrics[f'train_{loss_name}_loss'].append(loss_value)
                
                # Update progress bar
                train_loop.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mae': f'{criterion.last_losses["mae"]:.4f}',
                    'mse': f'{criterion.last_losses["mse"]:.4f}'
                })
                
                # Clean up memory
                del predictions, loss
                if use_amp:
                    torch.cuda.empty_cache()
            
            # Validation loop
            model.eval()
            val_losses = []
            val_metrics = defaultdict(list)
            
            with torch.no_grad():
                val_loop = tqdm(val_loader, desc='Validation')
                for batch_features, batch_targets in val_loop:
                    batch_features = batch_features.float().to(device)
                    batch_targets = batch_targets.float().to(device)
                    
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        predictions = model(batch_features)
                        loss = criterion(predictions, batch_targets)
                    
                    val_losses.append(loss.item())
                    for loss_name, loss_value in criterion.last_losses.items():
                        val_metrics[f'val_{loss_name}_loss'].append(loss_value)
                    
                    # Update validation progress bar
                    val_loop.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'mae': f'{criterion.last_losses["mae"]:.4f}',
                        'mse': f'{criterion.last_losses["mse"]:.4f}'
                    })
                    
                    # Clean up memory
                    del predictions, loss
                    if use_amp:
                        torch.cuda.empty_cache()
            
            # Calculate average losses
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            # Calculate average metrics
            train_metric_avgs = {name: np.mean(values) for name, values in train_metrics.items()}
            val_metric_avgs = {name: np.mean(values) for name, values in val_metrics.items()}
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Log metrics
            if wandb.run is not None:
                wandb.log({
                    'finetune/loss/train': avg_train_loss,
                    'finetune/loss/val': avg_val_loss,
                    'finetune/learning_rate': optimizer.param_groups[0]['lr'],
                    'finetune/metrics/train_mae': train_metric_avgs["train_mae_loss"],
                    'finetune/metrics/train_mse': train_metric_avgs["train_mse_loss"],
                    'finetune/metrics/val_mae': val_metric_avgs["val_mae_loss"],
                    'finetune/metrics/val_mse': val_metric_avgs["val_mse_loss"],
                    'finetune/epoch': epoch
                })
            
            # Print epoch summary
            print(f'\nEpoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {avg_train_loss:.4f} (MAE: {train_metric_avgs["train_mae_loss"]:.4f}, '
                  f'MSE: {train_metric_avgs["train_mse_loss"]:.4f})')
            print(f'Val Loss: {avg_val_loss:.4f} (MAE: {val_metric_avgs["val_mae_loss"]:.4f}, '
                  f'MSE: {val_metric_avgs["val_mse_loss"]:.4f})')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_epoch = epoch
                best_state = {k: v.cpu().clone().detach() for k, v in model.state_dict().items()}
                patience_counter = 0
                
                # Save best model state
                if trial_dir is not None:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': best_state,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': best_loss,
                        'metrics': {
                            'train': train_metric_avgs,
                            'val': val_metric_avgs
                        }
                    }, trial_dir / 'best_finetune_model.pth')
                    print(f"\nSaved new best model at epoch {epoch+1} with val_loss: {best_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'\nEarly stopping triggered after {epoch+1} epochs')
                    break
            
            # Generate prediction plots every 10 epochs
            if epoch % 10 == 0 and plots_dir is not None:
                model.eval()
                with torch.no_grad():
                    # Get a batch of validation data for plotting
                    val_features, val_targets = next(iter(val_loader))
                    val_features = val_features.float().to(device)
                    val_targets = val_targets.float().to(device)
                    
                    # Generate predictions
                    predictions = model(val_features)
                    
                    # Plot and log
                    plot_predictions(
                        predictions=predictions,
                        targets=val_targets,
                        joint_names=config['joint_angle_columns'],
                        model_name='transformer',
                        plots_dir=plots_dir,
                        epoch=epoch,
                        phase='val'
                    )
        
        # Generate final prediction plots
        model.eval()
        with torch.no_grad():
            val_features, val_targets = next(iter(val_loader))
            val_features = val_features.float().to(device)
            val_targets = val_targets.float().to(device)
            predictions = model(val_features)
            
            final_metrics = plot_predictions(
                predictions=predictions,
                targets=val_targets,
                joint_names=config['joint_angle_columns'],
                model_name='transformer',
                plots_dir=plots_dir,
                epoch='final',
                phase='val'
            )
            
            if wandb.run is not None:
                wandb.log({
                    'final/metrics': final_metrics
                })
        
        return best_loss, best_epoch, model
        
    except Exception as e:
        print(f"Error during finetuning: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return float('inf'), -1, model

if __name__ == "__main__":
    total_start = time.time()
    print("Starting model training...")
    
    # Create necessary directories at startup
    models_dir = Path('models')
    models_dir.mkdir(parents=True, exist_ok=True)
    trials_dir = models_dir / 'trials'
    trials_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path('plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("\nCreated directory structure:")
    print(f"- {models_dir}")
    print(f"- {trials_dir}")
    print(f"- {plots_dir}")
    
    # Check GPU and set device
    device, print_gpu_memory = check_gpu()
    print("\nInitial GPU Memory Stats:")
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    # Initialize wandb
    if os.environ.get('WANDB_DISABLED', '').lower() != 'true':
        wandb.login()
        print("Initialized wandb")
    
    # Base configuration
    base_config = {
        'data_path': "Z:/Divya/TEMP_transfers/toAni/BPN_P9LT_P9RT_flyCoords.csv",
        'batch_size': 64,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'initial_epochs': 10,
        'patience': 25,
        'pretrain_epochs': 100,
        'finetune_epochs': 150,
        'weight_decay': 0.01,
        'use_windows': True,  # Enable windowed data processing
        'window_size': 50,    # This will be overridden by optuna trials
        'model_params': {
            'input_size': 10,  # 10 velocity features: 4 z-vel, 4 x-vel, 2 combined
            'hidden_sizes': [256, 256],
            'dropout': 0.1,
            'nhead': 8,
            'num_layers': 4,
            'output_size': 18  # Changed to 18 flexion angles
        },
        'joint_angle_columns': [
            'L1A_flex', 'L1B_flex', 'L1C_flex',
            'L2A_flex', 'L2B_flex', 'L2C_flex',
            'L3A_flex', 'L3B_flex', 'L3C_flex',
            'R1A_flex', 'R1B_flex', 'R1C_flex',
            'R2A_flex', 'R2B_flex', 'R2C_flex',
            'R3A_flex', 'R3B_flex', 'R3C_flex'
        ],
        'velocity_features': [
            'z_vel', 'z_vel_ma5', 'z_vel_ma10', 'z_vel_ma20',
            'x_vel', 'x_vel_ma5', 'x_vel_ma10', 'x_vel_ma20',
            'velocity_magnitude', 'xz_velocity'
        ],
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }
    
    # List of models to train
    models_to_train = ['unsupervised_transformer']
    
    # Number of trials for hyperparameter optimization
    n_trials = 10
    
    # Results dictionary to store best trials for each model
    results = {}
    
    # Train each model
    for model_name in models_to_train:
        print(f"\n{'='*80}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*80}\n")
        
        # Create study for this model
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        )
        
        # Run optimization based on model type
        if model_name == 'unsupervised_transformer':
            study.optimize(
                lambda trial: objective_unsupervised(
                    trial, base_config, trial.number + 1, n_trials
                ),
                n_trials=n_trials,
                catch=(Exception,)
            )
        else:
            study.optimize(
                lambda trial: objective(
                    trial, model_name, base_config, trial.number + 1, n_trials
                ),
                n_trials=n_trials,
                catch=(Exception,)
            )
        
        # Store results
        results[model_name] = {
            'best_value': study.best_value,
            'best_params': study.best_trial.params
        }
        
        # Clear GPU memory between models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log completion of model training
        print(f"\nCompleted training {model_name}")
        print(f"Best validation loss: {study.best_value:.4f}")
        print("Best hyperparameters:")
        for param, value in study.best_trial.params.items():
            print(f"  {param}: {value}")
    
    # Print final results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"Best validation loss: {result['best_value']:.4f}")
        print("Best hyperparameters:")
        for param, value in result['best_params'].items():
            print(f"  {param}: {value}")
    
    total_end = time.time()
    print(f"\nTotal time taken for all models: {total_end - total_start:.2f} seconds")