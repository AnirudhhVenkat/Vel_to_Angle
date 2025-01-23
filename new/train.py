import time
from tqdm import tqdm
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wandb
from pathlib import Path
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

from models import base_lstm, deep_lstm, transformer, tcn
from models.unsupervised_transformer import UnsupervisedTransformerModel, pretrain_transformer, finetune_transformer
from utils.data import prepare_data, calculate_angular_velocities
from utils.metrics import calculate_metrics

wandb.login()

def plot_predictions(predictions, targets, model_name, epoch, section_size=600):
    """Plot predictions vs targets for each joint angle, with each trial as a separate subplot."""
    try:
        joint_names = [
            'L2B_rot', 'R3A_rot', 'R3A_flex', 
            'R1B_rot', 'R2B_rot', 'L3A_rot'
        ]
        
        # Convert inputs to numpy if they're tensors
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        
        # Create plots directory
        plots_dir = Path('plots')
        plots_dir.mkdir(parents=True, exist_ok=True)
        model_plots_dir = plots_dir / model_name
        model_plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate number of complete 600-frame sections
        total_timesteps = len(predictions)
        num_sections = total_timesteps // section_size
        if total_timesteps % section_size > 0:
            num_sections += 1
        
        print(f"\nPlotting predictions:")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Section size: {section_size}")
        print(f"Number of sections: {num_sections}")
        
        plot_paths = []
        
        # Create a separate figure for each joint angle
        for joint_idx, joint_name in enumerate(joint_names):
            # Calculate appropriate figure size based on number of sections
            fig_height = min(3 * num_sections, 25)  # Increase height from 15 to 25
            fig = plt.figure(figsize=(12, fig_height))
            fig.suptitle(f'{joint_name}\nModel: {model_name}', y=1.02)
            
            # Plot each section as a subplot
            for section in range(num_sections):
                start_idx = section * section_size
                end_idx = min((section + 1) * section_size, total_timesteps)
                
                # Skip if section is too small
                if end_idx - start_idx < 10:
                    continue
                
                # Create subplot
                ax = plt.subplot(num_sections, 1, section + 1)
                
                # Get data for this section
                section_predictions = predictions[start_idx:end_idx, joint_idx]
                section_targets = targets[start_idx:end_idx, joint_idx]
                
                # Print section info for debugging
                print(f"\nSection {section + 1}:")
                print(f"Start index: {start_idx}")
                print(f"End index: {end_idx}")
                print(f"Section length: {len(section_predictions)}")
                
                # Plot with reduced points for clarity
                stride = max(1, len(section_predictions) // 100)  # Plot at most 100 points per section
                x_values = np.arange(0, len(section_predictions), stride)
                
                # Plot with reduced points
                ax.plot(x_values, section_targets[::stride], 'b-', label='Target', alpha=0.7, linewidth=1.5)
                ax.plot(x_values, section_predictions[::stride], 'r-', label='Prediction', alpha=0.7, linewidth=1.5)
                
                # Calculate metrics for this section
                mse = np.mean((section_predictions - section_targets) ** 2)
                mae = np.mean(np.abs(section_predictions - section_targets))
                r2 = stats.pearsonr(section_predictions, section_targets)[0] ** 2
                
                # Add title and metrics
                ax.set_title(f'Trial {section + 1} (MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f})')
                ax.set_xlabel('Frame')
                ax.set_ylabel('Angle (degrees)')
                ax.grid(True, alpha=0.3)
                
                # Only show legend for first subplot
                if section == 0:
                    ax.legend()
                
                # Set consistent axis limits
                y_min = min(section_targets.min(), section_predictions.min())
                y_max = max(section_targets.max(), section_predictions.max())
                y_range = y_max - y_min
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
                
                # Set x-axis ticks to show actual frame numbers
                num_ticks = 5
                tick_positions = np.linspace(0, len(section_predictions), num_ticks, dtype=int)
                ax.set_xticks(tick_positions)
                ax.set_xticklabels([f'{x+start_idx}' for x in tick_positions])
            
            # Adjust layout and save
            plt.tight_layout()
            plot_path = model_plots_dir / f'predictions_epoch_{epoch}_{joint_name}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plot_paths.append(str(plot_path))
            plt.close(fig)
        
        # Save final prediction plots
        if epoch == -1:  # This indicates it's the final prediction plot
            final_plots_dir = plots_dir / 'final_predictions'
            final_plots_dir.mkdir(parents=True, exist_ok=True)
            
            for path in plot_paths:
                final_path = final_plots_dir / Path(path).name.replace('predictions_epoch_-1', f'{model_name}_final')
                shutil.copy2(path, final_path)
        
        # Log to wandb if available
        if wandb.run is not None:
            for path in plot_paths:
                wandb.log({
                    f"predictions/{Path(path).stem}": wandb.Image(path),
                    'epoch': epoch if epoch != -1 else 'final'
                })
        
        print(f"Saved prediction plots to {model_plots_dir}/")
        return plot_paths
    
    except Exception as e:
        print(f"\nFailed to generate prediction plots:")
        print(f"Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

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
        
        # Create trial directory
        models_dir = Path('models')
        models_dir.mkdir(parents=True, exist_ok=True)
        trials_dir = models_dir / 'trials'
        trials_dir.mkdir(parents=True, exist_ok=True)
        trial_dir = trials_dir / run_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb
        if wandb.run is None:
            wandb.init(
                project="joint_angle_prediction",
                name=run_name,
                config=config,
                reinit=True
            )
        
        # Train model
        train_loader, val_loader, test_loader = prepare_data(config)
        if not train_loader or not val_loader or not test_loader:
            raise ValueError("Data loaders are empty or not initialized properly.")
        
        model_classes = {
            'base_lstm': base_lstm.LSTMModel,
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
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        initial_epochs = config['initial_epochs']
        patience = config['patience']
        best_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        best_epoch = 0
        
        for epoch in range(config['num_epochs']):
            model.train()
            train_loss = 0.0
            
            if print_gpu_memory and epoch == 0:
                print(f"\nAt start of first epoch:")
                print_gpu_memory()
            
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                       desc=f'Epoch {epoch+1}/{config["num_epochs"]}')
            
            for batch_idx, (inputs, targets) in pbar:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                if print_gpu_memory and batch_idx == 0 and epoch == 0:
                    print(f"\nAfter first batch transfer:")
                    print_gpu_memory()
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                if wandb.run is not None:
                    wandb.log({
                        'train/batch_loss': loss.item(),
                        'train/batch': batch_idx + epoch * len(train_loader)
                    })
                
                # Clear memory after each batch
                del inputs, targets, outputs, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if print_gpu_memory and epoch == 0:
                print(f"\nAt end of first epoch:")
                print_gpu_memory()
            
            avg_train_loss = train_loss / len(train_loader)
            
            model.eval()
            val_loss = 0.0
            val_metrics = defaultdict(float)
            num_batches = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    batch_metrics = calculate_metrics(outputs, targets)
                    for k, v in batch_metrics.items():
                        val_metrics[k] += v
                    num_batches += 1
            
            avg_val_loss = val_loss / num_batches
            for k in val_metrics:
                val_metrics[k] /= num_batches
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                epochs_no_improve = 0
                print(f"\nNew best model found at epoch {epoch+1} with validation loss: {best_loss:.4f}")
                # Save the best model state
                model_save_path = trial_dir / 'best_model.pth'
                torch.save({
                    'model_state_dict': best_model_state,
                    'config': config,
                    'best_loss': best_loss,
                    'best_epoch': best_epoch,
                    'metrics': {
                        'test_loss': avg_val_loss,
                        'val_best_loss': best_loss,
                        'training_best_epoch': best_epoch,
                        'total_epochs': epoch + 1
                    }
                }, model_save_path)
                print(f"Saved best model to: {model_save_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience and epoch >= initial_epochs:
                    print('Early stopping!')
                    break
        
            scheduler.step(avg_val_loss)
        
        if best_model_state is None:
            raise ValueError("No valid model state was saved during training.")
        
        model.load_state_dict(best_model_state)
        
        print("\nGenerating final predictions and plots...")
        model.eval()
        test_predictions = []
        test_targets = []
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                test_predictions.append(outputs.cpu().numpy())
                test_targets.append(targets.cpu().numpy())

        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss}')

        test_predictions = np.concatenate(test_predictions, axis=0)
        test_targets = np.concatenate(test_targets, axis=0)
        
        print(f"\nBest model from epoch {best_epoch+1} for {model_name} (val_loss: {best_loss:.4f})")
        
        plot_paths = plot_predictions(test_predictions, test_targets, f"{model_name}_best", best_epoch)
        
        if plot_paths:
            for path in plot_paths:
                wandb.log({f"final_predictions_{model_name}": wandb.Image(path)})
        
        # Save configuration details
        config_path = trial_dir / 'model_config.txt'
        with open(config_path, 'w') as f:
            f.write(f"Run Name: {run_name}\n")
            f.write(f"Model Type: {model_name}\n")
            f.write(f"Hidden Sizes: {config['model_params']['hidden_sizes']}\n")
            f.write(f"Dropout: {config['model_params']['dropout']}\n")
            if model_name == 'transformer':
                f.write(f"Number of Heads: {config['model_params']['nhead']}\n")
                f.write(f"Number of Layers: {config['model_params']['num_layers']}\n")
            elif model_name == 'tcn':
                f.write(f"Kernel Size: {config['model_params']['kernel_size']}\n")
            f.write(f"Learning Rate: {config['learning_rate']}\n")
            f.write(f"Batch Size: {config['batch_size']}\n")
            f.write(f"\nTraining Results:\n")
            f.write(f"Best Loss: {best_loss}\n")
            f.write(f"Best Epoch: {best_epoch}\n")
            f.write(f"Test Loss: {test_loss}\n")
            f.write(f"Total Epochs: {epoch + 1}\n")
        
        # Save prediction plots
        plots_dir = trial_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        if plot_paths:
            for i, path in enumerate(plot_paths):
                plot_name = f'predictions_section_{i+1}.png'
                shutil.copy2(path, plots_dir / plot_name)
            print(f"Saved prediction plots to: {plots_dir}")
        
        print(f"\nSaved trial model and config to: {trial_dir}")
        wandb.finish()

        # Save the best model at the end of the trial
        if best_model_state is not None:
            final_model_save_path = trial_dir / f'final_best_model_trial.pth'
            torch.save({
                'model_state_dict': best_model_state,
                'config': config,
                'best_loss': best_loss,
                'best_epoch': best_epoch,
                'metrics': {
                    'test_loss': avg_val_loss,
                    'val_best_loss': best_loss,
                    'training_best_epoch': best_epoch,
                    'total_epochs': epoch + 1
                }
            }, final_model_save_path)
            print(f"Saved final best model for trial to: {final_model_save_path}")

        return best_loss
    
    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

def train_unsupervised_transformer(config):
    """Train an unsupervised transformer model with pretraining and finetuning."""
    try:
        device = config['device']
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create descriptive run name
        hidden_size = max(config['model_params']['hidden_sizes'])
        model_desc = f"unsupervised_transformer_h{hidden_size}_d{config['model_params']['dropout']}_head{config['model_params']['nhead']}_l{config['model_params']['num_layers']}"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{model_desc}_{timestamp}"
        
        # Create trial directory
        models_dir = Path('models')
        models_dir.mkdir(parents=True, exist_ok=True)
        trials_dir = models_dir / 'trials'
        trials_dir.mkdir(parents=True, exist_ok=True)
        trial_dir = trials_dir / run_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb
        wandb.init(
            project="joint_angle_prediction",
            name=run_name,
            config=config,
            reinit=True
        )
        
        # Enhanced data loading settings
        config['num_workers'] = min(8, os.cpu_count())
        config['pin_memory'] = True
        config['persistent_workers'] = True
        config['prefetch_factor'] = 2
        
        # Load and prepare data
        train_loader, val_loader, test_loader = prepare_data(config)
        if not train_loader or not val_loader or not test_loader:
            raise ValueError("Data loaders are empty or not initialized properly")
        
        # Initialize pretraining model
        model = UnsupervisedTransformerModel(
            input_size=config['model_params']['input_size'],
            hidden_size=hidden_size,
            nhead=config['model_params']['nhead'],
            num_layers=config['model_params']['num_layers'],
            output_size=config['model_params']['input_size'],  # For reconstruction
            dropout=config['model_params']['dropout']
        ).to(device)
        
        # Initialize criterion and optimizer for pretraining
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=True
        )
        
        # Pretraining phase
        print("\nStarting pretraining phase...")
        best_pretrain_loss, best_pretrain_epoch = pretrain_transformer(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            config=config,
            num_epochs=config['pretrain_epochs']
        )
        
        print(f"\nPretraining completed with best loss: {best_pretrain_loss:.4f} at epoch {best_pretrain_epoch}")
        
        # Save pretrained weights and clean up pretraining model
        print("\nSaving pretrained model state...")
        pretrained_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"Pretrained state keys: {pretrained_state.keys()}")

        # Save pretrained model immediately after creating pretrained_state
        print("\nSaving pretrained model...")
        pretrained_path = trial_dir / 'pretrained_model.pth'
        torch.save({
            'model_state_dict': pretrained_state,
            'config': config,
            'best_loss': best_pretrain_loss,
            'best_epoch': best_pretrain_epoch
        }, pretrained_path)
        print(f"Saved pretrained model to: {pretrained_path}")

        del model, optimizer, criterion
        torch.cuda.empty_cache()

        # Initialize finetuning model with same architecture but different output size
        print("\nInitializing finetuning model...")
        finetune_model = UnsupervisedTransformerModel(
            input_size=config['model_params']['input_size'],
            hidden_size=hidden_size,
            nhead=config['model_params']['nhead'],
            num_layers=config['model_params']['num_layers'],
            output_size=6,  # For joint angle prediction
            dropout=config['model_params']['dropout']
        ).to(device)

        # Transfer pretrained weights (excluding output layer)
        print("\nTransferring pretrained weights...")
        pretrained_dict = {k: v for k, v in pretrained_state.items() 
                          if 'output_projection' not in k}
        print(f"Keys to transfer: {pretrained_dict.keys()}")
        print(f"Target model keys: {finetune_model.state_dict().keys()}")

        missing_keys = finetune_model.load_state_dict(pretrained_dict, strict=False)
        print("\nTransferred pretrained weights:")
        print(f"Missing keys: {missing_keys.missing_keys}")
        print(f"Unexpected keys: {missing_keys.unexpected_keys}")

        # Verify weight transfer
        print("\nVerifying weight transfer...")
        for name, param in finetune_model.named_parameters():
            if 'output_projection' not in name:
                pretrained_param = pretrained_dict.get(name)
                if pretrained_param is not None:
                    if not torch.equal(param.cpu(), pretrained_param):
                        print(f"Warning: Weights for {name} do not match!")
                    else:
                        print(f"Successfully transferred weights for {name}")
                else:
                    print(f"Warning: No pretrained weights found for {name}")

        del pretrained_state, pretrained_dict
        torch.cuda.empty_cache()
        
        # Initialize optimizer and criterion for finetuning
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            finetune_model.parameters(),
            lr=config['learning_rate'],
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=True
        )
        
        # Finetuning phase
        best_finetune_loss, best_finetune_epoch, final_model = finetune_transformer(
            model=finetune_model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=config['finetune_epochs'],
            patience=config['patience'],
            gradient_accumulation_steps=4
        )
        
        if final_model is None:
            raise ValueError("Finetuning failed to return a valid model")
        
        print(f"\nFinetuning completed with best loss: {best_finetune_loss:.4f} at epoch {best_finetune_epoch}")
        
        # Generate predictions
        final_model.eval()
        test_predictions = []
        test_targets = []
        window_size = config['window_size']  # 200 frames
        chunk_size = 600  # Desired chunk size

        print(f"\nGenerating predictions:")
        print(f"Window size: {window_size}")
        print(f"Chunk size: {chunk_size}")

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            # Process data in windows of 200 frames
            for i in range(0, len(test_loader.dataset), window_size):
                # Get a window of data
                end_idx = min(i + window_size, len(test_loader.dataset))
                window_inputs = test_loader.dataset[i:end_idx][0]
                window_targets = test_loader.dataset[i:end_idx][1]
                
                # Only process complete windows
                if len(window_inputs) == window_size:
                    window_inputs = window_inputs.to(device, non_blocking=True)
                    outputs = final_model(window_inputs)
                    test_predictions.append(outputs.cpu())
                    test_targets.append(window_targets)
                
                # Clean up memory
                del outputs
                torch.cuda.empty_cache()

        # Concatenate all predictions and targets
        test_predictions = torch.cat(test_predictions, dim=0).numpy()
        test_targets = torch.cat(test_targets, dim=0).numpy()

        print(f"\nPrediction shape: {test_predictions.shape}")
        print(f"Target shape: {test_targets.shape}")

        # Organize predictions into 600-frame chunks for plotting
        total_frames = len(test_predictions)
        num_chunks = total_frames // chunk_size
        if total_frames % chunk_size > 0:
            num_chunks += 1

        print(f"Total frames: {total_frames}")
        print(f"Number of chunks: {num_chunks}")

        # Generate plots
        trial_info = wandb.run.name if wandb.run else "trial"
        plot_paths = plot_predictions(test_predictions, test_targets, f"unsupervised_transformer_{trial_info}", -1)
        
        # Log metrics
        metrics = {
            'test/mse': np.mean((test_predictions - test_targets) ** 2),
            'test/mae': np.mean(np.abs(test_predictions - test_targets)),
            'test/r2': np.mean([stats.pearsonr(test_predictions[:, i], test_targets[:, i])[0] ** 2 
                              for i in range(test_predictions.shape[1])]),
            'training/best_pretrain_loss': best_pretrain_loss,
            'training/best_finetune_loss': best_finetune_loss,
            'training/best_pretrain_epoch': best_pretrain_epoch,
            'training/best_finetune_epoch': best_finetune_epoch,
            'training/total_epochs': best_pretrain_epoch + best_finetune_epoch
        }
        
        if plot_paths:
            for i, path in enumerate(plot_paths):
                metrics[f'plots/final_predictions_{i+1}'] = wandb.Image(path)
        
        wandb.log(metrics)
        
        # Save both pretrained and finetuned models
        print("\nSaving models for this trial...")
        finetuned_path = trial_dir / 'finetuned_model.pth'

        # Save finetuned model
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'config': config,
            'best_loss': best_finetune_loss,
            'best_epoch': best_finetune_epoch,
            'metrics': metrics
        }, finetuned_path)

        # Save the best finetuned model at the end of the trial
        final_finetuned_path = trial_dir / f'final_best_finetuned_model_trial.pth'
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'config': config,
            'best_loss': best_finetune_loss,
            'best_epoch': best_finetune_epoch,
            'metrics': metrics
        }, final_finetuned_path)
        print(f"Saved final best finetuned model for trial to: {final_finetuned_path}")
        
        # Clean up only after all saving operations are complete
        del final_model, test_predictions, test_targets
        torch.cuda.empty_cache()
        
        # Save configuration details
        config_path = trial_dir / 'model_config.txt'
        with open(config_path, 'w') as f:
            f.write(f"Run Name: {run_name}\n")
            f.write(f"Hidden Size: {hidden_size}\n")
            f.write(f"Number of Heads: {config['model_params']['nhead']}\n")
            f.write(f"Number of Layers: {config['model_params']['num_layers']}\n")
            f.write(f"Dropout: {config['model_params']['dropout']}\n")
            f.write(f"Learning Rate: {config['learning_rate']}\n")
            f.write(f"Batch Size: {config['batch_size']}\n")
            f.write(f"\nTraining Results:\n")
            f.write(f"Best Pretrain Loss: {best_pretrain_loss}\n")
            f.write(f"Best Pretrain Epoch: {best_pretrain_epoch}\n")
            f.write(f"Best Finetune Loss: {best_finetune_loss}\n")
            f.write(f"Best Finetune Epoch: {best_finetune_epoch}\n")
            f.write(f"Test MSE: {metrics['test/mse']}\n")
            f.write(f"Test MAE: {metrics['test/mae']}\n")
            f.write(f"Test R2: {metrics['test/r2']}\n")
        
        # Save prediction plots
        plots_dir = trial_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        if plot_paths:
            for i, path in enumerate(plot_paths):
                plot_name = f'predictions_section_{i+1}.png'
                shutil.copy2(path, plots_dir / plot_name)
            print(f"Saved prediction plots to: {plots_dir}")
        
        print(f"\nSaved trial models and config to: {trial_dir}")
        wandb.finish()
        return best_finetune_loss
        
    except Exception as e:
        print(f"Error during unsupervised transformer training: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        wandb.finish()
        return float('inf')

def objective_unsupervised(trial, base_config, trial_num, total_trials):
    """Objective function for optimizing unsupervised transformer hyperparameters."""
    try:
        # Define hyperparameter search space with better ranges
        config = base_config.copy()
        config['batch_size'] = trial.suggest_categorical('batch_size', [32, 64, 128])  # Smaller batches for better stability
        config['learning_rate'] = trial.suggest_float('learning_rate', 5e-5, 1e-3, log=True)  # Lower learning rates
        config['dropout'] = trial.suggest_float('dropout', 0.1, 0.4)  # Reduced max dropout
        
        # Define model-specific parameters with better architectures
        hidden_sizes = []
        for i in range(3):  # 3 hidden layers
            hidden_sizes.append(trial.suggest_categorical(f'hidden_size_{i}', 
                [256, 384, 512, 768]))  # Larger hidden sizes for better representation
        
        # Transformer-specific parameters
        nhead = trial.suggest_categorical('nhead', [8, 16])  # More attention heads
        num_layers = trial.suggest_int('num_layers', 3, 6)  # More layers for better feature extraction
        
        # Add positional encoding type
        pos_encoding = trial.suggest_categorical('pos_encoding', ['learned', 'sinusoidal'])
        
        # Add new hyperparameters for training
        config['warmup_epochs'] = trial.suggest_int('warmup_epochs', 3, 8)
        config['gradient_clip'] = trial.suggest_float('gradient_clip', 0.5, 2.0)
        
        # Base model parameters
        base_params = {
            'input_size': 15,
            'hidden_sizes': hidden_sizes,
            'dropout': config['dropout'],
            'nhead': nhead,
            'num_layers': num_layers,
            'pos_encoding': pos_encoding
        }
        
        # Model parameters for pretraining
        config['model_params'] = base_params.copy()
        config['model_params']['output_size'] = 15  # Same as input for reconstruction
        
        # Add pretraining specific parameters
        config['pretrain_params'] = {
            'warmup_epochs': config['warmup_epochs'],
            'gradient_clip': config['gradient_clip'],
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),
            'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.1),
            'layer_norm_eps': trial.suggest_float('layer_norm_eps', 1e-7, 1e-5, log=True)
        }
        
        print(f"\nTrial {trial_num}/{total_trials}")
        print(f"\nUNSUPERVISED TRANSFORMER Configuration:")
        print("Architecture:")
        print(f"  Hidden sizes: {base_params['hidden_sizes']}")
        print(f"  Attention heads: {base_params['nhead']}")
        print(f"  Layers: {base_params['num_layers']}")
        print(f"  Positional encoding: {base_params['pos_encoding']}")
        print(f"  Dropout: {base_params['dropout']}")
        print("\nTraining:")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Learning rate: {config['learning_rate']}")
        print(f"  Warmup epochs: {config['warmup_epochs']}")
        print(f"  Gradient clip: {config['gradient_clip']}")
        print(f"  Weight decay: {config['pretrain_params']['weight_decay']}")
        print(f"  Label smoothing: {config['pretrain_params']['label_smoothing']}")
        
        final_val_loss = train_unsupervised_transformer(config)
        
        if final_val_loss is None:
            raise ValueError("Training returned None")
        
        return final_val_loss
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')

def pretrain_transformer(model, dataloader, criterion, optimizer, device, config, num_epochs=10, patience=10):
    """Enhanced pretraining with better optimization and monitoring."""
    model.train()
    best_loss = float('inf')
    best_epoch = -1
    best_model_state = model.state_dict()
    start_time = time.time()
    patience_counter = 0
    
    # Ensure minimum number of epochs and steps
    num_epochs = max(num_epochs, 1)  # Minimum epochs
    total_steps = len(dataloader) * num_epochs
    
    # Initialize scheduler with safety checks
    if total_steps < 100:  # If too few steps, use simpler scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        use_onecycle = False
    else:
        warmup_epochs = min(config['pretrain_params']['warmup_epochs'], num_epochs // 3)
        warmup_steps = len(dataloader) * warmup_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['learning_rate'],
            total_steps=total_steps,
            pct_start=warmup_steps/total_steps,
            anneal_strategy='cos',
            cycle_momentum=False
        )
        use_onecycle = True
    
    # Initialize AMP scaler
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    print(f"Training with early stopping (patience={patience} epochs)")
    print(f"Total steps: {total_steps}, Using OneCycle: {use_onecycle}")
    
    epoch_pbar = tqdm(range(num_epochs), desc='Pretraining Epochs')
    
    for epoch in epoch_pbar:
        epoch_start = time.time()  # Initialize epoch start time
        model.train()
        total_loss = 0.0
        num_samples = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Get inputs
            if isinstance(batch, (tuple, list)):
                inputs = batch[0]
            else:
                inputs = batch
            inputs = inputs.to(device, non_blocking=True)
            batch_size = inputs.size(0)
            
            # Forward pass with mixed precision
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):  # Updated autocast syntax
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)
                    if config['pretrain_params']['label_smoothing'] > 0:
                        loss = loss * (1 - config['pretrain_params']['label_smoothing'])
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                if config['pretrain_params']['gradient_clip'] > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        config['pretrain_params']['gradient_clip']
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                if config['pretrain_params']['label_smoothing'] > 0:
                    loss = loss * (1 - config['pretrain_params']['label_smoothing'])
                loss.backward()
                if config['pretrain_params']['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        config['pretrain_params']['gradient_clip']
                    )
                optimizer.step()
            
            # Update scheduler if using OneCycle
            if use_onecycle:
                scheduler.step()
            
            # Update metrics (use full batch loss)
            total_loss += loss.item() * batch_size
            num_samples += batch_size
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update progress bar
            avg_loss = total_loss / num_samples
            epoch_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # Clean up
            del outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate epoch metrics correctly
        avg_loss = total_loss / num_samples
        epoch_time = time.time() - epoch_start  # Calculate epoch time
        
        # Update scheduler if using ReduceLROnPlateau
        if not use_onecycle:
            scheduler.step(avg_loss)
        
        # Update best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            status = "(New best)"
            patience_counter = 0
        else:
            status = f"(No improvement: {patience_counter+1})"
            patience_counter += 1
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Loss: {avg_loss:.4f} {status}")
        print(f"Time: {epoch_time:.1f}s")
        print(f"Learning rate: {current_lr:.2e}")
        
        # Log epoch metrics
        if wandb.run is not None:
            wandb.log({
                'pretrain/epoch': epoch,
                'pretrain/loss': avg_loss,
                'pretrain/best_loss': best_loss,
                'pretrain/learning_rate': current_lr,
                'pretrain/epoch_time': epoch_time,
                'epoch': epoch
            })
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model from epoch {best_epoch+1}")
        print(f"Best loss: {best_loss:.4f}")
    
    return best_loss, best_epoch

def finetune_transformer(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100, patience=10, gradient_accumulation_steps=1):
    """Enhanced finetuning with better optimization and monitoring."""
    try:
        print("\nInitializing finetuning...")
        model.train()
        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = -1
        epochs_without_improvement = 0
        start_time = time.time()
        
        # Initialize loss functions
        mse_criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()
        
        # Initialize AMP scaler
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Initialize scheduler
        total_steps = (len(train_loader) // gradient_accumulation_steps) * num_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'],
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            cycle_momentum=False
        )
        
        print(f"Starting finetuning with early stopping (patience={patience} epochs)")
        print(f"Using gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Total optimization steps: {total_steps}")
        
        epoch_pbar = tqdm(range(num_epochs), desc='Finetuning Epochs')
        
        for finetune_epoch in epoch_pbar:
            model.train()
            train_mse = 0.0
            train_mae = 0.0
            num_samples = 0
            optimizer.zero_grad(set_to_none=True)
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                batch_size = inputs.size(0)
                
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    outputs = model(inputs)
                    mse_loss = mse_criterion(outputs, targets)
                    mae = mae_criterion(outputs, targets)
                    loss = mse_loss / gradient_accumulation_steps
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update metrics (use full batch loss, not scaled by gradient accumulation)
                train_mse += mse_loss.item() * batch_size
                train_mae += mae.item() * batch_size
                num_samples += batch_size
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                
                del outputs, mse_loss, mae, loss
                torch.cuda.empty_cache()
            
            # Compute average training metrics correctly
            avg_train_mse = train_mse / num_samples
            avg_train_mae = train_mae / num_samples
            
            # Validation phase
            model.eval()
            val_mse = 0.0
            val_mae = 0.0
            val_samples = 0
            
            chunk_size = 32
            with torch.no_grad():
                for i in range(0, len(val_loader.dataset), chunk_size):
                    end_idx = min(i + chunk_size, len(val_loader.dataset))
                    chunk_inputs = val_loader.dataset[i:end_idx][0]
                    chunk_targets = val_loader.dataset[i:end_idx][1]
                    chunk_size = chunk_inputs.size(0)
                    
                    chunk_inputs = chunk_inputs.to(device, non_blocking=True)
                    chunk_targets = chunk_targets.to(device, non_blocking=True)
                    
                    outputs = model(chunk_inputs)
                    mse = mse_criterion(outputs, chunk_targets)
                    mae = mae_criterion(outputs, chunk_targets)
                    
                    val_mse += mse.item() * chunk_size
                    val_mae += mae.item() * chunk_size
                    val_samples += chunk_size
                    
                    del outputs, mse, mae, chunk_inputs, chunk_targets
                    torch.cuda.empty_cache()
            
            # Compute average validation metrics correctly
            avg_val_mse = val_mse / val_samples
            avg_val_mae = val_mae / val_samples
            
            epoch_pbar.set_postfix({
                'train_mse': f'{avg_train_mse:.4f}',
                'train_mae': f'{avg_train_mae:.4f}',
                'val_mse': f'{avg_val_mse:.4f}',
                'val_mae': f'{avg_val_mae:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            if avg_val_mse < best_val_loss:
                best_val_loss = avg_val_mse
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = finetune_epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if wandb.run is not None:
                wandb.log({
                    'finetune/epoch': finetune_epoch,
                    'finetune/train_mse': avg_train_mse,
                    'finetune/train_mae': avg_train_mae,
                    'finetune/val_mse': avg_val_mse,
                    'finetune/val_mae': avg_val_mae,
                    'finetune/best_val_mse': best_val_loss,
                    'finetune/learning_rate': optimizer.param_groups[0]['lr']
                })
            
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {finetune_epoch+1} epochs")
                break
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            total_time = time.time() - start_time
            print(f"\nFinetuning completed in {total_time:.1f}s")
            print(f"Best validation MSE: {best_val_loss:.4f} at epoch {best_epoch+1}")
        
        return best_val_loss, best_epoch, model
    
    except Exception as e:
        print(f"\nError in finetune_transformer:")
        print(f"Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None, None, None

if __name__ == "__main__":
    total_start = time.time()
    print("Starting hyperparameter optimization...")
    
    # Check GPU and set device
    device, print_gpu_memory = check_gpu()
    print("\nInitial GPU Memory Stats:")
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    # Base configuration
    base_config = {
        'data_path': "Z:/Divya/TEMP_transfers/toAni/BPN_P9LT_P9RT_flyCoords.csv",
        'pretrain_path': "Z:/Divya/TEMP_transfers/toAni/BPN_P9LT_P9RT_flyCoords.csv",
        'window_size': 200,
        'split_ratio': 0.8,
        'test_size': 7200,
        'augment': False,
        'noise_level': 0.1,
        'n_folds': 5,
        'device': device,
        'initial_epochs': 120,
        'patience': 15,
        'num_epochs': 100,
        'pretrain_epochs': 50,
        'finetune_epochs': 100,
        # GPU optimizations
        'batch_size': 64,
        'num_workers': min(8, os.cpu_count()),
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 2,
        # Training optimizations
        'gradient_accumulation_steps': 4,
        'warmup_epochs': 5,
        'weight_decay': 0.01,
        'label_smoothing': 0.1
    }

    # Run optimization for each model
    model_names = ['unsupervised_transformer']
    total_models = len(model_names)
    total_trials = 10
    total_iterations = total_trials * total_models
    completed_iterations = 0
    
    for model_idx, model_name in enumerate(model_names, 1):
        model_start = time.time()
        print(f"\nOptimizing {model_name} ({model_idx}/{total_models})...")
        
        # Create a study object for this model
        study = optuna.create_study(
            study_name=f"{model_name}_optimization",
            direction="minimize",
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Run optimization
        if model_name == 'unsupervised_transformer':
            study.optimize(
                lambda trial: objective_unsupervised(trial, base_config, 
                                                   trial.number + 1, total_trials),
                n_trials=total_trials,
                timeout=86400  # 24 hour timeout
            )
        else:
            study.optimize(
                lambda trial: objective(trial, model_name, base_config, 
                                     trial.number + 1, total_trials),
                n_trials=total_trials,
                timeout=86400  # 24 hour timeout
            )
        
        completed_iterations += total_trials
        elapsed = time.time() - total_start
        avg_time_per_iter = elapsed / completed_iterations
        remaining_iterations = total_iterations - completed_iterations
        eta = avg_time_per_iter * remaining_iterations
        
        print("\nOverall Progress:")
        print(f"Models completed: {model_idx}/{total_models}")
        print(f"Total trials completed: {completed_iterations}/{total_iterations}")
        print(f"Elapsed time: {elapsed:.2f} seconds")
        print(f"Estimated time remaining: {eta:.2f} seconds")
        print(f"Average time per iteration: {avg_time_per_iter:.2f} seconds")
        
        print(f"\nCompleted optimization for {model_name}.")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value: {study.best_value}")
        print(f"Best parameters: {study.best_params}")
        
        model_end = time.time()
        print(f"Time taken for {model_name}: {model_end - model_start:.2f} seconds")
    
    total_end = time.time()
    print(f"\nTotal time taken for optimization: {total_end - total_start:.2f} seconds")