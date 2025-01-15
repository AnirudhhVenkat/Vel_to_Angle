import time
from tqdm import tqdm
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn  # Add this for nn.MSELoss
from torch.utils.data import DataLoader, TensorDataset  # Add these for data loading
import wandb
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
from models import base_lstm, deep_lstm, transformer, tcn
from utils.data import prepare_data
from utils.metrics import calculate_metrics
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd
def plot_predictions(predictions, targets, model_name, epoch):
    """Create and save plots for model predictions"""
    output_dir = Path(f'plots/{model_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy if tensors
    predictions = predictions if isinstance(predictions, np.ndarray) else predictions.numpy()
    targets = targets if isinstance(targets, np.ndarray) else targets.numpy()
    
    # Split data into 600-point segments
    segment_length = 600
    num_segments = len(predictions) // segment_length
    plot_paths = []
    
    for segment in range(num_segments):
        start_idx = segment * segment_length
        end_idx = (segment + 1) * segment_length
        
        # Create the plot filename for this segment
        plot_filename = f'prediction_epoch_{epoch+1}_segment_{segment+1}.png'
        plot_path = output_dir / plot_filename
        
        # Plot each joint angle for this segment
        fig, axes = plt.subplots(6, 3, figsize=(15, 20))
        fig.suptitle(f'{model_name} Predictions - Epoch {epoch+1} - Segment {segment+1}')
        
        for i in range(18):
            row = i // 3
            col = i % 3
            axes[row, col].plot(targets[start_idx:end_idx, i], label='Ground Truth', alpha=0.7)
            axes[row, col].plot(predictions[start_idx:end_idx, i], label='Prediction', alpha=0.7)
            axes[row, col].set_title(f'Joint Angle {i+1}')
            axes[row, col].legend()
        
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        if plot_path.exists():
            plot_paths.append(str(plot_path))
    
    # Return list of plot paths for wandb logging
    return plot_paths if plot_paths else None

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

def objective(trial, model_name, base_config, trial_number, total_trials):
    """Add trial progress tracking"""
    print(f"\nTrial {trial_number}/{total_trials}")
    # Define hyperparameter search space
    hp_config = {
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
    }
    
    # Model specific hyperparameters
    if model_name == 'transformer':
        hp_config.update({
            'hidden_sizes': [
                trial.suggest_int(f'hidden_size_{i}', 32, 512, step=32)
                for i in range(3)
            ],
            'nhead': trial.suggest_categorical('nhead', [4, 8, 16]),
            'num_layers': trial.suggest_int('num_layers', 2, 6)
        })
    else:
        hp_config.update({
            'hidden_sizes': [
                trial.suggest_int(f'hidden_size_{i}', 64, 512, step=64)
                for i in range(3)
            ],
        })
    
    # Update config with trial hyperparameters
    config = base_config.copy()
    config.update({
        'batch_size': hp_config['batch_size'],
        'learning_rate': hp_config['learning_rate'],
        'model_params': {
            'input_size': 6,
            'hidden_sizes': hp_config['hidden_sizes'],
            'output_size': 18,
            'dropout': hp_config['dropout'],
            **(({'nhead': hp_config['nhead'], 'num_layers': hp_config['num_layers']} 
                if model_name == 'transformer' else {}))
        }
    })
    
    # Train model with current hyperparameters
    try:
        if model_name == 'unsupervised_transformer':
            final_val_loss = train_unsupervised_transformer(config)
        else:
            final_val_loss = train_model(model_name, config)
        return final_val_loss
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')

def train_model(model_name, config):
    try:
        start_time = time.time()
        # Create descriptive name with hyperparameters
        model_desc = get_model_name(model_name, config)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{model_desc}_{timestamp}"
        
        # Initialize wandb with new run
        wandb.init(project="joint_angle_prediction", name=run_name, config=config, reinit=True)
        
        # Create model save directory with clear architecture-based organization
        save_dir = Path('best_models')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_save_path = save_dir / f'{model_desc}_best.pth'
        
        # Load and prepare data with validation
        try:
            train_loader, val_loader, test_loader = prepare_data(config)
            print(f"Data loaded successfully. Training batches: {len(train_loader)}, "
                  f"Validation batches: {len(val_loader)}, Test batches: {len(test_loader)}")
        except Exception as e:
            print(f"Failed to prepare data: {str(e)}")
            raise

        # Validate batch size
        if config['batch_size'] > len(train_loader.dataset):
            raise ValueError(f"Batch size ({config['batch_size']}) larger than dataset "
                           f"size ({len(train_loader.dataset)})")

        # Initialize model
        model_classes = {
            'base_lstm': base_lstm.LSTMModel,
            'deep_lstm': deep_lstm.DeepLSTM,
            'transformer': transformer.TransformerModel,
            'tcn': tcn.TCNModel
        }
        
        # Update transformer model initialization to set norm_first to False
        if model_name == 'transformer':
            config['model_params']['norm_first'] = False
        
        # Remove norm_first from config before model initialization
        if model_name == 'transformer':
            if 'norm_first' in config['model_params']:
                del config['model_params']['norm_first']
        
        # Initialize model with validation
        try:
            model = model_classes[model_name](**config['model_params'])
            model = model.to(config['device'])
            print(f"Model initialized successfully on {config['device']}")
        except Exception as e:
            print(f"Failed to initialize model: {str(e)}")
            raise
        
        # Use DataParallel for multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
        
        # Initialize loss function and optimizer
        criterion = nn.MSELoss()  # Replace CustomLoss with simple MSE
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        # Training loop
        initial_epochs = config['initial_epochs']
        patience = config['patience']
        best_loss = float('inf')
        epochs_no_improve = 0

        # Store best model state
        best_model_state = None
        best_epoch = 0
        
        # Add gradient clipping
        max_grad_norm = 1.0
        
        # Initialize validation loss with infinity
        best_loss = float('inf')
        
        # Perform initial validation before training
        model.eval()
        initial_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(config['device']), targets.to(config['device'])
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                initial_val_loss += loss.item()
        initial_val_loss /= len(val_loader)
        print(f"Initial validation loss: {initial_val_loss}")
        best_loss = initial_val_loss
        
        total_batches = len(train_loader) * config['num_epochs']
        global_start_time = time.time()
        total_processed_batches = 0
        
        for epoch in range(config['num_epochs']):
            epoch_start = time.time()
            # Training phase
            model.train()
            train_loss = 0.0
            num_batches = len(train_loader)
            
            # Add progress bar for batches
            pbar = tqdm(enumerate(train_loader), total=num_batches, 
                       desc=f'Epoch {epoch+1}/{config["num_epochs"]}')
            
            for batch_idx, (inputs, targets) in pbar:
                inputs, targets = inputs.to(config['device']), targets.to(config['device'])
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                train_loss += loss.item()
                
                # Update total progress
                total_processed_batches += 1
                elapsed_total = time.time() - global_start_time
                batches_per_second = total_processed_batches / elapsed_total
                remaining_batches = total_batches - total_processed_batches
                eta_seconds = remaining_batches / batches_per_second
                
                # Update progress bar with both epoch and total progress
                elapsed = time.time() - epoch_start
                epoch_progress = (batch_idx + 1) / num_batches
                eta = elapsed / epoch_progress * (1 - epoch_progress)
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'epoch_eta': f'{eta:.1f}s',
                    'total_eta': str(timedelta(seconds=int(eta_seconds)))
                })
            
            # Calculate average training loss
            train_loss /= num_batches
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_metrics = defaultdict(float)
            num_batches = 0
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(config['device']), targets.to(config['device'])
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    # Calculate all metrics for this batch
                    batch_metrics = calculate_metrics(outputs, targets)
                    for k, v in batch_metrics.items():
                        val_metrics[k] += v
                    num_batches += 1
            
            # Average the metrics
            val_loss /= num_batches
            for k in val_metrics:
                val_metrics[k] /= num_batches
            
            # Log all metrics
            metrics_log = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr,
                **{f'val_{k}': v for k, v in val_metrics.items()}  # This logs val_mse, val_rmse, val_mae, val_mape, val_r2, val_dtw
            }
            wandb.log(metrics_log)
            
            # Print comprehensive metrics
            print(f'Epoch {epoch+1}:')
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            print('Validation Metrics:')
            for k, v in val_metrics.items():
                print(f'  {k}: {v:.4f}')
            print(f'Learning Rate: {current_lr:.6f}')
            
            # Use combined metric for early stopping
            validation_score = val_loss * (1 + (1 - val_metrics['r2']))  # Penalize both high loss and low R²
            
            if validation_score < best_loss:
                best_loss = validation_score
                epochs_no_improve = 0
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                    'config': config,
                    'timestamp': timestamp
                }
                best_epoch = epoch
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print('Early stopping!')
                    break
        
        # Load best model state for final evaluation
        model.load_state_dict(best_model_state['model_state_dict'])
        
        # Test
        model.eval()
        test_loss = 0.0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(config['device']), targets.to(config['device'])
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss}')

        # Concatenate all predictions and targets
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Now save the model and create plots after training is complete
        # torch.save(best_model_state, model_save_path)
        print(f"\nSaved best model from epoch {best_epoch+1} for {model_name} (val_loss: {best_loss:.4f})")
        
        # Create and save plots
        plot_paths = plot_predictions(all_predictions, all_targets, model_name, best_epoch)
        
        # Save final results
        results_dir = Path(f'C:/Users/bidayelab/vel_to_angle_project/new/results/{model_name}')
        results_dir.mkdir(parents=True, exist_ok=True)
        # np.save(results_dir / 'predictions.npy', all_predictions)
        # np.save(results_dir / 'targets.npy', all_targets)

        # Log final metrics to wandb, only include plots if they exist
        final_metrics = {'test_loss': test_loss}
        if plot_paths:
            for i, path in enumerate(plot_paths):
                final_metrics[f'final_predictions_segment_{i+1}'] = wandb.Image(path)
        wandb.log(final_metrics)
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

    return best_loss

def train_unsupervised_transformer(config):
    from models.unsupervised_transformer import (
        UnsupervisedTransformerModel,
        pretrain_transformer,
        finetune_transformer
    )
    
    device = config['device']
    
    if 'pretrain_path' not in config:
        print("No pretrain_path specified in config, using data_path")
        config['pretrain_path'] = config['data_path']
    
    # 1. Load unlabeled data
    try:
        df_unlabeled = pd.read_csv(config['pretrain_path'])
        print(f"Loaded unlabeled data with shape: {df_unlabeled.shape}")
    except Exception as e:
        print(f"Failed to load unlabeled data: {e}")
        return float('inf')
    
    # Convert to PyTorch tensor
    try:
        X_unlabeled = torch.tensor(df_unlabeled.values, dtype=torch.float32)
        unlabeled_dataset = TensorDataset(X_unlabeled)
        unlabeled_loader = DataLoader(
            unlabeled_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True,
            num_workers=0
        )
        print(f"Created unlabeled DataLoader with {len(unlabeled_loader)} batches")
    except Exception as e:
        print(f"Failed to create DataLoader: {e}")
        return float('inf')
    
    # 2. Initialize model for unsupervised pretraining
    try:
        model = UnsupervisedTransformerModel(**config['model_params']).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # 3. Pretrain in unsupervised manner
        pretrain_loss = pretrain_transformer(
            model, 
            unlabeled_loader, 
            criterion, 
            optimizer, 
            device, 
            num_epochs=5
        )
        
        # 4. Finetune on labeled data
        train_loader, val_loader, _ = prepare_data(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        
        final_val_loss = finetune_transformer(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            num_epochs=5
        )
        
        return final_val_loss
        
    except Exception as e:
        print(f"Error during model training: {e}")
        return float('inf')

# ...existing code...

if __name__ == "__main__":
    total_start = time.time()
    print("Starting hyperparameter optimization...")
    
    # Set up GPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Base configuration
    base_config = {
        'data_path': "Z:/Divya/TEMP_transfers/toAni/BPN_P9LT_P9RT_flyCoords.csv",
        'pretrain_path': "Z:/Divya/TEMP_transfers/toAni/BPN_P9LT_P9RT_flyCoords.csv",  # Add this line
        'window_size': 200,
        'split_ratio': 0.8,
        'test_size': 7200,
        'augment': False,
        'noise_level': 0.1,
        'n_folds': 5,
        'device': device,
        # Removed loss_alpha as it's no longer needed
        'initial_epochs': 120,
        'patience': 15,
        'num_epochs': 100,  # Reduced for faster trials
    }

    # Run optimization for each model
    model_names = ['base_lstm', 
                   'deep_lstm', 
                   'transformer', 
                   'tcn',
                   'unsupervised_transformer']
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
        study.optimize(
            lambda trial: objective(trial, model_name, base_config, 
                                 trial.number + 1, total_trials),
            n_trials=total_trials,  # Number of trials to run
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
        print(f"Time elapsed: {timedelta(seconds=int(elapsed))}")
        print(f"Estimated time remaining: {timedelta(seconds=int(eta))}")
        
        # Print optimization results
        print(f"\nBest trial for {model_name}:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        
        # After finding best trial, train and log only the best model
        best_trial = study.best_trial
        best_config = base_config.copy()
        best_config.update({
            'batch_size': best_trial.params['batch_size'],
            'learning_rate': best_trial.params['learning_rate'],
            'model_params': {
                'input_size': 6,
                'hidden_sizes': [best_trial.params[f'hidden_size_{i}'] for i in range(3)],
                'output_size': 18,
                'dropout': best_trial.params['dropout'],
                **(({'nhead': best_trial.params['nhead'],
                     'num_layers': best_trial.params['num_layers']}
                    if model_name == 'transformer' else {}))
            }
        })
        
        # Initialize wandb for best model only
        model_desc = get_model_name(model_name, best_config)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{model_desc}_best_{timestamp}"
        
        with wandb.init(project="joint_angle_prediction", 
                       name=run_name, 
                       config=best_config) as run:
            train_model(model_name, best_config)

        # After each model completes, show progress
        elapsed = time.time() - total_start
        avg_time_per_model = elapsed / model_idx
        remaining_models = total_models - model_idx
        eta = avg_time_per_model * remaining_models
        
        print(f"\nProgress: {model_idx}/{total_models} models completed")
        print(f"Time elapsed: {elapsed/3600:.1f} hours")
        print(f"Estimated time remaining: {eta/3600:.1f} hours")
        
    total_time = time.time() - total_start
    print(f"\nHyperparameter optimization completed!")
    print(f"Total time taken: {total_time/3600:.1f} hours")