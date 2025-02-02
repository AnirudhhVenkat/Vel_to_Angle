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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

def log_feature_specific_metrics(predictions, targets, phase='train'):
    """Log metrics separately for positions, velocities, and joint angles."""
    if wandb.run is not None:
        if predictions.shape[-1] == 18:  # Pretraining mode
            # Split predictions and targets into positions and velocities
            pos_pred = predictions[..., :6]  # 6 TaG y-positions
            vel_pred = predictions[..., 6:]  # 12 velocity features
            pos_targ = targets[..., :6]
            vel_targ = targets[..., 6:]
            
            # Calculate metrics
            pos_mae = torch.mean(torch.abs(pos_pred - pos_targ))
            vel_mae = torch.mean(torch.abs(vel_pred - vel_targ))
            
            wandb.log({
                f'{phase}/position_mae': pos_mae.item(),
                f'{phase}/velocity_mae': vel_mae.item()
            })
        else:  # Finetuning mode (48 joint angles)
            # Calculate metrics for each leg type
            for leg_idx, leg in enumerate(['L1', 'R1', 'L2', 'R2', 'L3', 'R3']):
                start_idx = leg_idx * 8
                end_idx = start_idx + 8
                leg_pred = predictions[..., start_idx:end_idx]
                leg_targ = targets[..., start_idx:end_idx]
                
                # Calculate metrics for different angle types
                flex_mae = torch.mean(torch.abs(leg_pred[..., [0,3,5,7]] - leg_targ[..., [0,3,5,7]]))  # A_flex, B_flex, C_flex, D_flex
                rot_mae = torch.mean(torch.abs(leg_pred[..., [1,4,6]] - leg_targ[..., [1,4,6]]))      # A_rot, B_rot, C_rot
                abduct_mae = torch.mean(torch.abs(leg_pred[..., 2] - leg_targ[..., 2]))               # A_abduct
                
                wandb.log({
                    f'{phase}/{leg}_flex_mae': flex_mae.item(),
                    f'{phase}/{leg}_rot_mae': rot_mae.item(),
                    f'{phase}/{leg}_abduct_mae': abduct_mae.item()
                })

def plot_predictions(model, data_loader, feature_scaler, target_scaler, device, config, phase='test', max_samples=5):
    """Plot predictions vs actual values for a subset of samples."""
    model.eval()
    predictions = []
    targets = []
    inputs = []
    
    with torch.no_grad():
        for batch in data_loader:
            if len(predictions) >= max_samples:
                break
                
            batch_inputs, batch_targets = batch
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            batch_pred = model(batch_inputs)
            
            predictions.extend(batch_pred.cpu().numpy())
            targets.extend(batch_targets.cpu().numpy())
            inputs.extend(batch_inputs.cpu().numpy())
            
            if len(predictions) >= max_samples:
                predictions = predictions[:max_samples]
                targets = targets[:max_samples]
                inputs = inputs[:max_samples]
                break
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    inputs = np.array(inputs)
    
    # Create plots directory if it doesn't exist
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    
    if model.pretraining:  # Plotting pretraining results
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot TaG y-positions
        ax = axes[0]
        for i in range(6):
            ax.plot(predictions[0, :, i], label=f'Pred TaG{i+1}')
            ax.plot(targets[0, :, i], '--', label=f'True TaG{i+1}')
        ax.set_title('TaG Y-Positions')
        ax.legend()
        ax.grid(True)
        
        # Plot velocities
        ax = axes[1]
        for i in range(12):
            ax.plot(predictions[0, :, i+6], label=f'Pred Vel{i+1}')
            ax.plot(targets[0, :, i+6], '--', label=f'True Vel{i+1}')
        ax.set_title('Velocity Features')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'{phase}_pretraining_predictions.png')
        plt.close()
        
    else:  # Plotting finetuning results (joint angles)
        # Create separate plots for each leg
        for leg_idx, leg in enumerate(['L1', 'R1', 'L2', 'R2', 'L3', 'R3']):
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))
            start_idx = leg_idx * 8
            
            # Plot flexion angles
            ax = axes[0]
            flex_indices = [start_idx + i for i in [0,3,5,7]]  # A_flex, B_flex, C_flex, D_flex
            for i, angle in enumerate(['A_flex', 'B_flex', 'C_flex', 'D_flex']):
                ax.plot(predictions[0, :, flex_indices[i]], label=f'Pred {angle}')
                ax.plot(targets[0, :, flex_indices[i]], '--', label=f'True {angle}')
            ax.set_title(f'{leg} Flexion Angles')
            ax.legend()
            ax.grid(True)
            
            # Plot rotation angles
            ax = axes[1]
            rot_indices = [start_idx + i for i in [1,4,6]]  # A_rot, B_rot, C_rot
            for i, angle in enumerate(['A_rot', 'B_rot', 'C_rot']):
                ax.plot(predictions[0, :, rot_indices[i]], label=f'Pred {angle}')
                ax.plot(targets[0, :, rot_indices[i]], '--', label=f'True {angle}')
            ax.set_title(f'{leg} Rotation Angles')
            ax.legend()
            ax.grid(True)
            
            # Plot abduction angle
            ax = axes[2]
            abduct_idx = start_idx + 2  # A_abduct
            ax.plot(predictions[0, :, abduct_idx], label='Pred A_abduct')
            ax.plot(targets[0, :, abduct_idx], '--', label='True A_abduct')
            ax.set_title(f'{leg} Abduction Angle')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'{phase}_finetuning_{leg}_predictions.png')
        plt.close()
        
        # Create a summary plot with all legs
        plt.figure(figsize=(15, 10))
        for leg_idx, leg in enumerate(['L1', 'R1', 'L2', 'R2', 'L3', 'R3']):
            start_idx = leg_idx * 8
            plt.plot(predictions[0, :, start_idx], label=f'{leg} A_flex Pred')
            plt.plot(targets[0, :, start_idx], '--', label=f'{leg} A_flex True')
        plt.title('A_flex Angles for All Legs')
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / f'{phase}_finetuning_summary.png')
        plt.close()
    
    if wandb.run is not None:
        wandb.log({
            f'{phase}/predictions_plot': wandb.Image(str(plots_dir / f'{phase}_predictions.png')),
            f'{phase}/summary_plot': wandb.Image(str(plots_dir / f'{phase}_finetuning_summary.png'))
        })

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
        plot_predictions(model, test_loader, feature_scaler, target_scaler, device, config, phase='test')
        
        # Log final metrics
        log_wandb({
            'test/loss': test_loss,
            'training/total_epochs': config['num_epochs'],
            'training/best_epoch': best_epoch,
            'training/best_val_loss': best_loss,
        })
        
        # Save final model and results
        final_results = {
            'model_state_dict': best_model_state,
            'config': config,
            'metrics': {
                'train_loss': train_loss,
                'val_loss': best_loss,
                'best_epoch': best_epoch,
                'total_epochs': config['num_epochs'],
                'test_loss': test_loss
            },
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
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'dropout': 0.1,
        'num_epochs': 100,
        'patience': 15,
        'initial_epochs': 10,
        'pretrain_epochs': 100,
        'finetune_epochs': 150,
        
        # Data processing options
        'use_windows': True,
        'window_size': 50,
        
        'model_params': {
            'input_size': 15,  # Updated: 6 TaG positions + 9 velocity features
            'hidden_sizes': [256, 256],
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
        
        # Updated feature list
        'tag_position_features': [
            'L-F-TaG_y', 'R-F-TaG_y',
            'L-M-TaG_y', 'R-M-TaG_y',
            'L-H-TaG_y', 'R-H-TaG_y'
        ],
        
        'velocity_features': [
            # Z-velocity features
            'z_vel', 'z_vel_ma5', 'z_vel_ma10',
            # X-velocity features
            'x_vel', 'x_vel_ma5', 'x_vel_ma10',
            # Combined velocities
            'velocity_magnitude', 'xz_velocity'
        ],
        
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        
        # Initialize model with correct input size (18 features: 6 positions + 12 velocities)
        model = UnsupervisedTransformerModel(
            input_size=18,
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
            'finetune_epoch': finetune_epoch,
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler
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

def prepare_data(config):
    """Load and prepare data for training."""
    print("\nPreparing data...")
    
    # Load data
    df = pd.read_csv(config['data_path'])
    print(f"Loaded data with shape: {df.shape}")
    
    # Calculate moving averages for velocities
    raw_velocities = ['x_vel', 'y_vel', 'z_vel']
    ma_windows = [5, 10, 20]
    
    # Ensure raw velocities exist
    if not all(col in df.columns for col in raw_velocities):
        raise ValueError(f"Raw velocity columns {raw_velocities} not found in data")
    
    # Calculate moving averages within each trial
    trial_length = 600  # After filtering (frames 400-1000)
    num_trials = len(df) // trial_length
    
    for vel in raw_velocities:
        for window in ma_windows:
            ma_col = f'{vel}_ma{window}'
            ma_values = []
            
            for trial in range(num_trials):
                start_idx = trial * trial_length
                end_idx = (trial + 1) * trial_length
                trial_data = df[vel].iloc[start_idx:end_idx]
                
                # Calculate moving average for this trial
                ma = trial_data.rolling(window=window, center=True, min_periods=1).mean()
                ma_values.extend(ma)
            
            df[ma_col] = ma_values
    
    # Define feature columns
    velocity_features = [
        'x_vel_ma5', 'y_vel_ma5', 'z_vel_ma5',      # Moving averages
        'x_vel_ma10', 'y_vel_ma10', 'z_vel_ma10',
        'x_vel_ma20', 'y_vel_ma20', 'z_vel_ma20',
        'x_vel', 'y_vel', 'z_vel'                    # Raw velocities
    ]
    
    position_features = [
        'L-F-TaG_y', 'R-F-TaG_y',  # Front legs
        'L-M-TaG_y', 'R-M-TaG_y',  # Middle legs
        'L-H-TaG_y', 'R-H-TaG_y'   # Hind legs
    ]
    
    # Define joint angle columns for each leg
    joint_angles = []
    for leg in ['L1', 'R1', 'L2', 'R2', 'L3', 'R3']:
        joint_angles.extend([
            f'{leg}A_flex', f'{leg}A_rot', f'{leg}A_abduct',
            f'{leg}B_flex', 'B_rot',
            f'{leg}C_flex', 'C_rot',
            f'{leg}D_flex'
        ])
    
    # Verify all required columns exist
    missing_cols = []
    for col in position_features + velocity_features + joint_angles:
        if col not in df.columns:
            missing_cols.append(col)
    
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")
    
    # Combine input features
    input_features = position_features + velocity_features
    print(f"\nInput features ({len(input_features)}):")
    print("Position features:", position_features)
    print("Velocity features:", velocity_features)
    
    print(f"\nOutput features ({len(joint_angles)}):")
    print("Joint angles:", joint_angles)
    
    # Extract features and targets
    X = df[input_features].values
    y = df[joint_angles].values
    
    # Create scalers
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    # Fit and transform the data
    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y)
    
    # Create windowed sequences if enabled
    if config.get('use_windows', True):
        window_size = config['window_size']
        X_windows = []
        y_windows = []
        
        # Create windows for each trial
        for trial in range(num_trials):
            start_idx = trial * trial_length
            end_idx = (trial + 1) * trial_length
            
            trial_X = X_scaled[start_idx:end_idx]
            trial_y = y_scaled[start_idx:end_idx]
            
            # Create windows within trial
            for i in range(0, trial_length - window_size + 1):
                X_windows.append(trial_X[i:i+window_size])
                y_windows.append(trial_y[i+window_size-1])  # Predict last frame
        
        X_windows = np.array(X_windows)
        y_windows = np.array(y_windows)
        print(f"\nCreated windowed sequences:")
        print(f"X shape: {X_windows.shape}")
        print(f"y shape: {y_windows.shape}")
        
        # Use windowed data
        X_scaled = X_windows
        y_scaled = y_windows
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_scaled, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    # Create dataloaders
    train_data = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_data = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_data = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(
        train_data,
        batch_size=config['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )
    test_loader = DataLoader(
        test_data,
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )
    
    print("\nData preparation completed:")
    print(f"Train set: {len(train_data)} samples")
    print(f"Val set: {len(val_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    
    return train_loader, val_loader, test_loader, feature_scaler, target_scaler

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
            'input_size': 18,  # 6 TaG y-positions + 12 velocity features
            'hidden_sizes': [256, 256],
            'output_size': 48,  # 8 angles per leg × 6 legs
            'dropout': 0.1,
            'nhead': 8,
            'num_layers': 4
        },
        # Input features
        'position_features': [
            'L-F-TaG_y', 'R-F-TaG_y',  # Front legs
            'L-M-TaG_y', 'R-M-TaG_y',  # Middle legs
            'L-H-TaG_y', 'R-H-TaG_y'   # Hind legs
        ],
        'velocity_features': [
            'x_vel_ma5', 'y_vel_ma5', 'z_vel_ma5',      # Moving averages
            'x_vel_ma10', 'y_vel_ma10', 'z_vel_ma10',
            'x_vel_ma20', 'y_vel_ma20', 'z_vel_ma20',
            'x_vel', 'y_vel', 'z_vel'                    # Raw velocities
        ],
        # Output features (joint angles)
        'joint_angles': [
            # Left Front Leg (L1)
            'L1A_flex', 'L1A_rot', 'L1A_abduct',
            'L1B_flex', 'L1B_rot',
            'L1C_flex', 'L1C_rot',
            'L1D_flex',
            # Right Front Leg (R1)
            'R1A_flex', 'R1A_rot', 'R1A_abduct',
            'R1B_flex', 'R1B_rot',
            'R1C_flex', 'R1C_rot',
            'R1D_flex',
            # Left Middle Leg (L2)
            'L2A_flex', 'L2A_rot', 'L2A_abduct',
            'L2B_flex', 'L2B_rot',
            'L2C_flex', 'L2C_rot',
            'L2D_flex',
            # Right Middle Leg (R2)
            'R2A_flex', 'R2A_rot', 'R2A_abduct',
            'R2B_flex', 'R2B_rot',
            'R2C_flex', 'R2C_rot',
            'R2D_flex',
            # Left Hind Leg (L3)
            'L3A_flex', 'L3A_rot', 'L3A_abduct',
            'L3B_flex', 'L3B_rot',
            'L3C_flex', 'L3C_rot',
            'L3D_flex',
            # Right Hind Leg (R3)
            'R3A_flex', 'R3A_rot', 'R3A_abduct',
            'R3B_flex', 'R3B_rot',
            'R3C_flex', 'R3C_rot',
            'R3D_flex'
        ],
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }
    
    # List of models to train
    models_to_train = ['unsupervised_transformer']
    
    # Number of trials for hyperparameter optimization
    n_trials = 1
    
    # Results dictionary to store best trials for each model
    results = {}
    
    # Train each model
    for model_name in models_to_train:
        print(f"\nTraining {model_name}...")
        
        # Create study for hyperparameter optimization
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=5
            )
        )
        
        # Run optimization
        study.optimize(
            lambda trial: objective_unsupervised(trial, base_config, len(study.trials), n_trials),
            n_trials=n_trials,
            timeout=None,
            catch=(Exception,)
        )
        
        # Print optimization results
        print("\nOptimization Results:")
        print(f"Number of finished trials: {len(study.trials)}")
        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        
        # Store results
        results[model_name] = {
            'best_value': trial.value,
            'best_params': trial.params,
            'n_trials': len(study.trials)
        }
    
    # Print final results
    print("\nFinal Results:")
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Best validation loss: {result['best_value']}")
        print("  Best parameters:")
        for key, value in result['best_params'].items():
            print(f"    {key}: {value}")
        print(f"  Number of trials: {result['n_trials']}")
    
    # Calculate total runtime
    total_time = time.time() - total_start
    print(f"\nTotal runtime: {timedelta(seconds=int(total_time))}")