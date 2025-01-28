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

class DTWLoss(nn.Module):
    """Dynamic Time Warping loss function using FastDTW."""
    def __init__(self):
        super(DTWLoss, self).__init__()
        self.l1_loss = nn.L1Loss()  # MAE loss
        self.mse_loss = nn.MSELoss()  # For metrics only
        self.batch_count = 0
        self.radius = 10  # Radius for FastDTW
    
    def compute_dtw(self, pred, target):
        """Compute DTW distance using FastDTW."""
        # Convert tensors to numpy for FastDTW
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Compute FastDTW
        distance, _ = fastdtw(pred_np, target_np, dist=euclidean, radius=self.radius)
        return torch.tensor(distance, device=pred.device) / (pred.shape[0] * pred.shape[1])
    
    def forward(self, pred, target):
        """
        Calculate combined loss between prediction and target sequences.
        Args:
            pred: Predicted sequences (batch_size, seq_len, features)
            target: Target sequences (batch_size, seq_len, features)
        Returns:
            Combined loss value (MAE + DTW)
        """
        # Calculate MAE loss on full batch
        mae_loss = self.l1_loss(pred, target)
        
        # Calculate MSE for metrics only
        mse_loss = self.mse_loss(pred, target)
        
        # For DTW, only use first sequence from batch
        dtw_loss = self.compute_dtw(pred[0], target[0])
        
        # Print progress occasionally
        self.batch_count += 1
        if self.batch_count % 100 == 0:
            print(f"\nProcessed {self.batch_count} batches in DTWLoss")
            print(f"Current losses - MAE: {mae_loss.item():.4f}, DTW: {dtw_loss.item():.4f}")
            print(f"MSE (metric): {mse_loss.item():.4f}")
        
        # Combine losses with equal weight on MAE and DTW
        final_loss = 0.5 * mae_loss + 0.5 * dtw_loss
        
        # Store individual losses for logging
        self.last_losses = {
            'mae': mae_loss.item(),
            'dtw': dtw_loss.item(),
            'mse': mse_loss.item(),  # Store MSE as a metric
            'total': final_loss.item()
        }
        
        return final_loss

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

def plot_predictions(predictions, targets, joint_names, model_name='model', use_windows=False, window_size=50):
    """Plot predictions vs targets for each joint, with trials of 600 frames each.
    
    Args:
        predictions: Model predictions (batch_size, num_joints) or (batch_size, seq_len, num_joints)
        targets: Target values (batch_size, num_joints) or (batch_size, seq_len, num_joints)
        joint_names: List of joint names
        model_name: Name of the model for plot titles
        use_windows: Whether the data is windowed
        window_size: Size of windows if use_windows is True
    """
    import matplotlib.pyplot as plt
    import wandb
    from pathlib import Path
    from datetime import datetime
    
    # Create plots directory if it doesn't exist
    plots_dir = Path('plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Convert to numpy if tensors
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    
    # Check if predictions/targets are 3D (batch_size, seq_len, num_joints)
    # or 2D (batch_size, num_joints)
    is_sequence = len(predictions.shape) == 3
    
    # For windowed data with stride=1, we need to reconstruct the full sequence
    if use_windows and is_sequence:
        # Initialize arrays for full sequence reconstruction
        seq_length = (predictions.shape[0] + window_size - 1)  # Total sequence length with stride=1
        full_predictions = np.zeros((seq_length, predictions.shape[-1]))
        full_targets = np.zeros((seq_length, targets.shape[-1]))
        counts = np.zeros(seq_length)  # For averaging overlapping predictions
        
        # Reconstruct sequences by averaging overlapping windows
        for i in range(predictions.shape[0]):
            full_predictions[i:i+window_size] += predictions[i]
            full_targets[i:i+window_size] += targets[i]
            counts[i:i+window_size] += 1
        
        # Average overlapping regions
        full_predictions = full_predictions / counts[:, np.newaxis]
        full_targets = full_targets / counts[:, np.newaxis]
        
        predictions = full_predictions
        targets = full_targets
        is_sequence = False  # After reconstruction, treat as 2D
    
    # Split into trials of 600 frames
    trial_length = 600
    if is_sequence:
        num_trials = predictions.shape[0]
        trial_predictions = predictions
        trial_targets = targets
    else:
        num_trials = len(predictions) // trial_length
        trial_predictions = predictions.reshape(num_trials, trial_length, -1)
        trial_targets = targets.reshape(num_trials, trial_length, -1)
    
    # Calculate metrics for each joint and trial
    metrics = {}
    for i, joint in enumerate(joint_names):
        joint_metrics = []
        for trial in range(num_trials):
            if is_sequence:
                trial_mae = np.mean(np.abs(trial_predictions[trial, :, i] - trial_targets[trial, :, i]))
                trial_mse = np.mean((trial_predictions[trial, :, i] - trial_targets[trial, :, i])**2)
            else:
                trial_data = slice(trial * trial_length, (trial + 1) * trial_length)
                trial_mae = np.mean(np.abs(predictions[trial_data, i] - targets[trial_data, i]))
                trial_mse = np.mean((predictions[trial_data, i] - targets[trial_data, i])**2)
            joint_metrics.append({'mae': trial_mae, 'mse': trial_mse})
        
        metrics[f'{joint}_mae'] = np.mean([m['mae'] for m in joint_metrics])
        metrics[f'{joint}_mse'] = np.mean([m['mse'] for m in joint_metrics])
    
    # Calculate overall metrics
    metrics['overall_mae'] = np.mean([metrics[f'{j}_mae'] for j in joint_names])
    metrics['overall_mse'] = np.mean([metrics[f'{j}_mse'] for j in joint_names])
    
    # Group joints by leg for better visualization
    legs = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']
    segments = ['A', 'B', 'C']
    
    # Create a figure for each leg
    for leg in legs:
        # Get all joints for this leg
        leg_joints = [j for j in joint_names if j.startswith(leg)]
        
        # Create subplots for each trial and segment
        fig_height = 3 * num_trials  # 3 inches per trial
        fig, axes = plt.subplots(num_trials, len(segments), figsize=(15, fig_height))
        fig.suptitle(f'{model_name} - {leg} Leg Flexion Angles')
        
        # Plot each trial
        for trial_idx in range(num_trials):
            for seg_idx, segment in enumerate(segments):
                joint = f'{leg}{segment}_flex'
                joint_idx = joint_names.index(joint)
                ax = axes[trial_idx, seg_idx]
                
                time_points = np.arange(trial_length)
                if is_sequence:
                    pred_data = trial_predictions[trial_idx, :, joint_idx]
                    target_data = trial_targets[trial_idx, :, joint_idx]
                else:
                    trial_data = slice(trial_idx * trial_length, (trial_idx + 1) * trial_length)
                    pred_data = predictions[trial_data, joint_idx]
                    target_data = targets[trial_data, joint_idx]
                
                # Calculate trial-specific metrics
                trial_mae = np.mean(np.abs(pred_data - target_data))
                trial_mse = np.mean((pred_data - target_data)**2)
                
                ax.plot(time_points, pred_data, label='Predicted', color='blue', alpha=0.7)
                ax.plot(time_points, target_data, label='Target', color='red', alpha=0.7)
                ax.set_title(f'{joint}\nTrial {trial_idx+1} (MAE: {trial_mae:.4f})')
                ax.set_xlabel('Frame')
                ax.set_ylabel('Angle (degrees)')
                ax.legend()
                ax.grid(True)
        
        plt.tight_layout()
        
        # Save the figure
        plot_path = plots_dir / f'{model_name}_{leg}_leg_{timestamp}.png'
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        print(f"Saved plot to: {plot_path}")
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                f'predictions/{leg}_leg': wandb.Image(str(plot_path))
            })
        
        plt.close()
        
    # Save metrics to a file
    metrics_path = plots_dir / f'{model_name}_metrics_{timestamp}.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Window settings: use_windows={use_windows}, window_size={window_size}\n\n")
        f.write("Metrics:\n")
        
        # Write metrics by leg
        for leg in legs:
            f.write(f"\n{leg} Leg:\n")
            leg_joints = [j for j in joint_names if j.startswith(leg)]
            for joint in leg_joints:
                f.write(f"  {joint}:\n")
                f.write(f"    MAE: {metrics[f'{joint}_mae']:.4f}\n")
                f.write(f"    MSE: {metrics[f'{joint}_mse']:.4f}\n")
        
        f.write(f"\nOverall Metrics:\n")
        f.write(f"  MAE: {metrics['overall_mae']:.4f}\n")
        f.write(f"  MSE: {metrics['overall_mse']:.4f}\n")
    
    print(f"Saved metrics to: {metrics_path}")
    
    # Log overall metrics to wandb
    if wandb.run is not None:
        wandb.log({
            'metrics/overall_mae': metrics['overall_mae'],
            'metrics/overall_mse': metrics['overall_mse']
        })
        
        # Log individual joint metrics
        for joint in joint_names:
            wandb.log({
                f'metrics/{joint}/mae': metrics[f'{joint}_mae'],
                f'metrics/{joint}/mse': metrics[f'{joint}_mse']
            })
    
    return metrics

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
            use_windows=config.get('use_windows', False),
            window_size=config.get('window_size', 50)
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
        
        l1_criterion = nn.L1Loss()
        dtw_criterion = DTWLoss().to(model.device)
        
        def combined_loss(pred, target, alpha=0.5):
            """Combine L1 and DTW losses."""
            l1_loss = l1_criterion(pred, target)
            dtw_loss = dtw_criterion(pred.unsqueeze(1) if len(pred.shape) == 2 else pred, 
                                   target.unsqueeze(1) if len(target.shape) == 2 else target)
            return l1_loss * (1 - alpha) + dtw_loss * alpha, l1_loss, dtw_loss
        
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
            train_l1_losses = []
            train_dtw_losses = []
            
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
                        loss, l1_loss, dtw_loss = combined_loss(outputs, batch_features)
                    else:
                        loss, l1_loss, dtw_loss = combined_loss(outputs, batch_targets)
                
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
                train_l1_losses.append(l1_loss.item())
                train_dtw_losses.append(dtw_loss.item())
            
            # Validation phase
            model.eval()
            val_losses = []
            val_l1_losses = []
            val_dtw_losses = []
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.float()
                    batch_targets = batch_targets.float()
                    
                    if torch.cuda.is_available():
                        batch_features = batch_features.cuda()
                        batch_targets = batch_targets.cuda()
                    
                    with torch.cuda.amp.autocast(device_type='cuda', enabled=use_amp):
                        outputs = model(batch_features)
                        if model.pretraining:
                            loss, l1_loss, dtw_loss = combined_loss(outputs, batch_features)
                        else:
                            loss, l1_loss, dtw_loss = combined_loss(outputs, batch_targets)
                    
                    val_losses.append(loss.item())
                    val_l1_losses.append(l1_loss.item())
                    val_dtw_losses.append(dtw_loss.item())
            
            avg_train_loss = np.mean(train_losses)
            avg_train_l1 = np.mean(train_l1_losses)
            avg_train_dtw = np.mean(train_dtw_losses)
            avg_val_loss = np.mean(val_losses)
            avg_val_l1 = np.mean(val_l1_losses)
            avg_val_dtw = np.mean(val_dtw_losses)
            
            # Update history
            history['pretrain_train_loss'].append(avg_train_loss)
            history['pretrain_val_loss'].append(avg_val_loss)
            history['pretrain_epochs'].append(epoch)
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    'pretrain/train_loss': avg_train_loss,
                    'pretrain/train_l1_loss': avg_train_l1,
                    'pretrain/train_dtw_loss': avg_train_dtw,
                    'pretrain/val_loss': avg_val_loss,
                    'pretrain/val_l1_loss': avg_val_l1,
                    'pretrain/val_dtw_loss': avg_val_dtw,
                    'pretrain/epoch': epoch,
                    'pretrain/learning_rate': optimizer.param_groups[0]['lr']
                })
            
            scheduler.step(avg_val_loss)
            
            print(f'Pretrain Epoch {epoch+1}:')
            print(f'  Train Loss: {avg_train_loss:.6f}')
            print(f'  Val Loss: {avg_val_loss:.6f}')
            
            # Save only the best pretrained model
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
            train_l1_losses = []
            train_dtw_losses = []
            
            for batch_features, batch_targets in tqdm(train_loader, desc=f'Finetune Epoch {epoch+1}'):
                batch_features = batch_features.float()
                batch_targets = batch_targets.float()
                
                if torch.cuda.is_available():
                    batch_features = batch_features.cuda()
                    batch_targets = batch_targets.cuda()
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(batch_features)
                    if model.pretraining:
                        loss, l1_loss, dtw_loss = combined_loss(outputs, batch_features)
                    else:
                        loss, l1_loss, dtw_loss = combined_loss(outputs, batch_targets)
                
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
                train_l1_losses.append(l1_loss.item())
                train_dtw_losses.append(dtw_loss.item())
            
            model.eval()
            val_losses = []
            val_l1_losses = []
            val_dtw_losses = []
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.float().to(device)
                    batch_targets = batch_targets.float().to(device)
                    
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        outputs = model(batch_features)
                        loss = dtw_criterion(outputs, batch_targets)
                    
                    val_losses.append(loss.item())
                    val_l1_losses.append(l1_loss.item())
                    val_dtw_losses.append(dtw_loss.item())
            
            avg_train_loss = np.mean(train_losses)
            avg_train_l1 = np.mean(train_l1_losses)
            avg_train_dtw = np.mean(train_dtw_losses)
            avg_val_loss = np.mean(val_losses)
            avg_val_l1 = np.mean(val_l1_losses)
            avg_val_dtw = np.mean(val_dtw_losses)
            
            # Update history
            history['finetune_train_loss'].append(avg_train_loss)
            history['finetune_val_loss'].append(avg_val_loss)
            history['finetune_epochs'].append(epoch)
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    'finetune/train_loss': avg_train_loss,
                    'finetune/train_l1_loss': avg_train_l1,
                    'finetune/train_dtw_loss': avg_train_dtw,
                    'finetune/val_loss': avg_val_loss,
                    'finetune/val_l1_loss': avg_val_l1,
                    'finetune/val_dtw_loss': avg_val_dtw,
                    'finetune/epoch': epoch,
                    'finetune/learning_rate': optimizer.param_groups[0]['lr'],
                    **{f'finetune/{k}': v for k, v in {
                        'train_mae_loss': avg_train_l1,
                        'train_dtw_loss': avg_train_dtw,
                        'train_mse_loss': avg_train_dtw
                    }.items()},
                    **{f'finetune/{k}': v for k, v in {
                        'val_mae_loss': avg_val_l1,
                        'val_dtw_loss': avg_val_dtw,
                        'val_mse_loss': avg_val_dtw
                    }.items()}
                })
            
            # Print epoch summary
            print(f'\nEpoch {epoch+1}/{config.get("finetune_epochs", 150)}:')
            print(f'Train Loss: {avg_train_loss:.4f} (MAE: {avg_train_l1:.4f}, '
                  f'DTW: {avg_train_dtw:.4f}, MSE: {avg_train_dtw:.4f})')
            print(f'Val Loss: {avg_val_loss:.4f} (MAE: {avg_val_l1:.4f}, '
                  f'DTW: {avg_val_dtw:.4f}, MSE: {avg_val_dtw:.4f})')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
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
        
        # Use the best finetuned model for final evaluation
        model.load_state_dict(best_finetune_state)
        model.eval()
        model.set_pretraining(False)
        
        # Generate and save final predictions plot
        metrics = plot_predictions(
            test_predictions, 
            test_targets, 
            joint_names=config['joint_angle_columns'],
            model_name=model_name,
            use_windows=config['use_windows'],
            window_size=config['window_size']
        )
        
        # Log final metrics
        log_wandb({
            'test/loss': test_loss,
            'test/metrics': metrics,
            'training/total_epochs': config['num_epochs'],
            'training/best_epoch': best_finetune_epoch,
            'training/best_val_loss': best_finetune_loss,
        })
        
        # Save final model and results
        final_results = {
            'model_state_dict': best_finetune_state,
            'pretrained_state_dict': best_pretrain_state,
            'config': config,
            'metrics': metrics,
            'history': history,
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler,
            'pretrain_loss': best_pretrain_loss,
            'pretrain_epoch': best_pretrain_epoch,
            'finetune_loss': best_finetune_loss,
            'finetune_epoch': best_finetune_epoch
        }
        
        torch.save(final_results, trial_dir / 'final_model.pth')
        
        if wandb.run is not None:
            wandb.finish()
        
        return model, metrics
        
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
    """Pretrain the transformer model using reconstruction loss."""
    # Ensure model is in pretraining mode
    model.set_pretraining(True)
    
    dtw_criterion = DTWLoss().to(device)
    
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
    
    # Lists to store losses for plotting
    train_losses = []
    val_losses = []
    epochs = []
    
    # Enable gradient scaler and AMP for mixed precision
    use_amp = torch.cuda.is_available() and hasattr(torch.cuda.amp, 'autocast')
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    try:
        for epoch in range(num_epochs):
            model.train()
            epoch_train_losses = []
            train_metrics = defaultdict(list)
            
            # Training loop with progress bar
            train_loop = tqdm(train_loader, desc=f'Pretrain Epoch {epoch+1}')
            for batch_idx, (batch_features, _) in enumerate(train_loop):
                batch_features = batch_features.float().to(device)
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(batch_features)
                    loss = dtw_criterion(outputs, batch_features)
                
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
                epoch_train_losses.append(loss.item())
                for loss_name, loss_value in dtw_criterion.last_losses.items():
                    train_metrics[f'train_{loss_name}_loss'].append(loss_value)
                
                # Update progress bar
                train_loop.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mae': f'{dtw_criterion.last_losses["mae"]:.4f}',
                    'dtw': f'{dtw_criterion.last_losses["dtw"]:.4f}',
                    'mse': f'{dtw_criterion.last_losses["mse"]:.4f}'
                })
                
                # Log to wandb every 50 batches
                if wandb.run is not None and batch_idx % 50 == 0:
                    wandb.log({
                        'pretrain/batch': batch_idx + epoch * len(train_loader),
                        'pretrain/train_loss': loss.item(),
                        'pretrain/train_mae_loss': dtw_criterion.last_losses['mae'],
                        'pretrain/train_dtw_loss': dtw_criterion.last_losses['dtw'],
                        'pretrain/train_mse': dtw_criterion.last_losses['mse']
                    })
                
                # Clean up memory
                del outputs, loss
                if scaler is not None:
                    torch.cuda.empty_cache()
            
            # Validation loop
            model.eval()
            epoch_val_losses = []
            val_metrics = defaultdict(list)
            
            with torch.no_grad():
                val_loop = tqdm(val_loader, desc='Validation')
                for batch_features, _ in val_loop:
                    batch_features = batch_features.float().to(device)
                    
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        outputs = model(batch_features)
                        loss = dtw_criterion(outputs, batch_features)
                    
                    # Store losses for logging
                    epoch_val_losses.append(loss.item())
                    for loss_name, loss_value in dtw_criterion.last_losses.items():
                        val_metrics[f'val_{loss_name}_loss'].append(loss_value)
                    
                    # Update validation progress bar
                    val_loop.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'mae': f'{dtw_criterion.last_losses["mae"]:.4f}',
                        'dtw': f'{dtw_criterion.last_losses["dtw"]:.4f}',
                        'mse': f'{dtw_criterion.last_losses["mse"]:.4f}'
                    })
            
            # Calculate average losses
            avg_train_loss = np.mean(epoch_train_losses)
            avg_val_loss = np.mean(epoch_val_losses)
            
            # Calculate average metrics
            train_metric_avgs = {name: np.mean(values) for name, values in train_metrics.items()}
            val_metric_avgs = {name: np.mean(values) for name, values in val_metrics.items()}
            
            # Store losses for plotting
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            epochs.append(epoch)
            
            # Create and log learning curves plot to wandb
            if wandb.run is not None:
                wandb.log({
                    "pretrain/train_loss": avg_train_loss,
                    "pretrain/val_loss": avg_val_loss,
                    "pretrain/train_mae": train_metric_avgs["train_mae_loss"],
                    "pretrain/train_dtw": train_metric_avgs["train_dtw_loss"],
                    "pretrain/train_mse": train_metric_avgs["train_mse_loss"],
                    "pretrain/val_mae": val_metric_avgs["val_mae_loss"],
                    "pretrain/val_dtw": val_metric_avgs["val_dtw_loss"],
                    "pretrain/val_mse": val_metric_avgs["val_mse_loss"],
                    "pretrain/epoch": epoch,
                    "pretrain/learning_rate": optimizer.param_groups[0]['lr']
                })
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Log metrics
            if wandb.run is not None:
                wandb.log({
                    'pretrain/epoch': epoch,
                    'pretrain/train_loss': avg_train_loss,
                    'pretrain/val_loss': avg_val_loss,
                    'pretrain/learning_rate': optimizer.param_groups[0]['lr'],
                    **{f'pretrain/{k}': v for k, v in train_metric_avgs.items()},
                    **{f'pretrain/{k}': v for k, v in val_metric_avgs.items()}
                })
            
            # Print epoch summary
            print(f'\nEpoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {avg_train_loss:.4f} (MAE: {train_metric_avgs["train_mae_loss"]:.4f}, '
                  f'DTW: {train_metric_avgs["train_dtw_loss"]:.4f}, MSE: {train_metric_avgs["train_mse_loss"]:.4f})')
            print(f'Val Loss: {avg_val_loss:.4f} (MAE: {val_metric_avgs["val_mae_loss"]:.4f}, '
                  f'DTW: {val_metric_avgs["val_dtw_loss"]:.4f}, MSE: {val_metric_avgs["val_mse_loss"]:.4f})')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_epoch = epoch
                best_state = model.state_dict()
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
        
        # Load best model state
        model.load_state_dict(best_state)
        return best_loss, best_epoch, best_state
    
    except Exception as e:
        print(f"Error during pretraining: {str(e)}")
        traceback.print_exc()
        return float('inf'), -1, None

def finetune_transformer(model, train_loader, val_loader, device, config, num_epochs=150, patience=10, trial_dir=None, plots_dir=None, run_name=None):
    """Finetune the transformer model for the target task."""
    dtw_criterion = DTWLoss().to(device)
    
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
    
    # Lists to store losses for plotting
    train_losses = []
    val_losses = []
    epochs = []
    
    # Enable gradient scaler and AMP for mixed precision
    use_amp = torch.cuda.is_available() and hasattr(torch.cuda.amp, 'autocast')
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    try:
        for epoch in range(num_epochs):
            model.train()
            epoch_train_losses = []
            train_metrics = defaultdict(list)
            
            # Training loop with progress bar
            train_loop = tqdm(train_loader, desc=f'Finetune Epoch {epoch+1}')
            for batch_idx, (batch_features, batch_targets) in enumerate(train_loop):
                batch_features = batch_features.float().to(device)
                batch_targets = batch_targets.float().to(device)
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(batch_features)
                    loss = dtw_criterion(outputs, batch_targets)
                
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
                epoch_train_losses.append(loss.item())
                for loss_name, loss_value in dtw_criterion.last_losses.items():
                    train_metrics[f'train_{loss_name}_loss'].append(loss_value)
                
                # Update progress bar
                train_loop.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mae': f'{dtw_criterion.last_losses["mae"]:.4f}',
                    'dtw': f'{dtw_criterion.last_losses["dtw"]:.4f}',
                    'mse': f'{dtw_criterion.last_losses["mse"]:.4f}'
                })
                
                # Log to wandb every 50 batches
                if wandb.run is not None and batch_idx % 50 == 0:
                    wandb.log({
                        'finetune/batch': batch_idx + epoch * len(train_loader),
                        'finetune/train_loss': loss.item(),
                        'finetune/train_mae_loss': dtw_criterion.last_losses['mae'],
                        'finetune/train_dtw_loss': dtw_criterion.last_losses['dtw'],
                        'finetune/train_mse': dtw_criterion.last_losses['mse']
                    })
                
                # Clean up memory
                del outputs, loss
                if scaler is not None:
                    torch.cuda.empty_cache()
            
            # Validation loop
            model.eval()
            epoch_val_losses = []
            val_metrics = defaultdict(list)
            
            with torch.no_grad():
                val_loop = tqdm(val_loader, desc='Validation')
                for batch_features, batch_targets in val_loop:
                    batch_features = batch_features.float().to(device)
                    batch_targets = batch_targets.float().to(device)
                    
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        outputs = model(batch_features)
                        loss = dtw_criterion(outputs, batch_targets)
                    
                    # Store losses for logging
                    epoch_val_losses.append(loss.item())
                    for loss_name, loss_value in dtw_criterion.last_losses.items():
                        val_metrics[f'val_{loss_name}_loss'].append(loss_value)
                    
                    # Update validation progress bar
                    val_loop.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'mae': f'{dtw_criterion.last_losses["mae"]:.4f}',
                        'dtw': f'{dtw_criterion.last_losses["dtw"]:.4f}',
                        'mse': f'{dtw_criterion.last_losses["mse"]:.4f}'
                    })
            
            # Calculate average losses
            avg_train_loss = np.mean(epoch_train_losses)
            avg_train_l1 = np.mean(train_metrics['train_mae_loss'])
            avg_train_dtw = np.mean(train_metrics['train_dtw_loss'])
            avg_val_loss = np.mean(epoch_val_losses)
            avg_val_l1 = np.mean(val_metrics['val_mae_loss'])
            avg_val_dtw = np.mean(val_metrics['val_dtw_loss'])
            
            # Store losses for plotting
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            epochs.append(epoch)
            
            # Create and log learning curves plot to wandb
            if wandb.run is not None:
                wandb.log({
                    "finetune/train_loss": avg_train_loss,
                    "finetune/val_loss": avg_val_loss,
                    "finetune/train_mae": train_metric_avgs["train_mae_loss"],
                    "finetune/train_dtw": train_metric_avgs["train_dtw_loss"],
                    "finetune/train_mse": train_metric_avgs["train_mse_loss"],
                    "finetune/val_mae": val_metric_avgs["val_mae_loss"],
                    "finetune/val_dtw": val_metric_avgs["val_dtw_loss"],
                    "finetune/val_mse": val_metric_avgs["val_mse_loss"],
                    "finetune/epoch": epoch,
                    "finetune/learning_rate": optimizer.param_groups[0]['lr']
                })
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            # Log metrics
            if wandb.run is not None:
                wandb.log({
                    'finetune/epoch': epoch,
                    'finetune/train_loss': avg_train_loss,
                    'finetune/val_loss': avg_val_loss,
                    'finetune/learning_rate': optimizer.param_groups[0]['lr'],
                    **{f'finetune/{k}': v for k, v in train_metrics.items()},
                    **{f'finetune/{k}': v for k, v in val_metrics.items()}
                })
            
            # Print epoch summary
            print(f'\nEpoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {avg_train_loss:.4f} (MAE: {avg_train_l1:.4f}, '
                  f'DTW: {avg_train_dtw:.4f})')
            print(f'Val Loss: {avg_val_loss:.4f} (MAE: {avg_val_l1:.4f}, '
                  f'DTW: {avg_val_dtw:.4f})')
            print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_epoch = epoch
                best_state = model.state_dict()
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
                            'train': train_metrics,
                            'val': val_metrics
                        }
                    }, trial_dir / 'best_finetune_model.pth')
                    print(f"\nSaved new best model at epoch {epoch+1} with val_loss: {best_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'\nEarly stopping triggered after {epoch+1} epochs')
                    break
        
        # Load best model state
        model.load_state_dict(best_state)
        return best_loss, best_epoch, model
    
    except Exception as e:
        print(f"Error during finetuning: {str(e)}")
        traceback.print_exc()
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