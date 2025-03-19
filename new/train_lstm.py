import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime
import json
import optuna
from optuna.trial import TrialState
import shutil
import pickle
import traceback
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from matplotlib.backends.backend_pdf import PdfPages
from scipy import signal
import random
from losses import DerivativeLoss
import os

# Import functions from data.py
from data import (
    set_all_seeds,
    TrialAwareSequenceDataset,
    TrialSampler,
    ZScoreScaler,
    filter_trial_frames,
    calculate_enhanced_features,
    get_available_data_path,
    calculate_psd_features,
    add_psd_features,
    prepare_data,
)

# Set seed for all random operations
def set_all_seeds(seed=42):
    """Set seeds for all random number generators for reproducibility."""
    random.seed(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"All random seeds set to {seed}")

# Call the function at the beginning
set_all_seeds(42)

class LSTMPredictor(nn.Module):
    """LSTM model for predicting joint angles from velocity features."""
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True  # Input shape: [batch, seq, features]
        )
        
        # Fully connected layer for output
        self.fc = nn.Linear(hidden_size, output_size)
        
        print("\nModel Architecture:")
        print(f"Input size: {input_size}")
        print(f"Hidden size: {hidden_size}")
        print(f"Number of layers: {num_layers}")
        print(f"Output size: {output_size}")
        print(f"Dropout: {dropout}")
        
    def forward(self, x):
        # Initialize hidden state
        batch_size = x.size(0)
        device = x.device
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        
        # Forward propagate LSTM
        # lstm_out shape: [batch_size, seq_len, hidden_size]
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Get the last output for each sequence
        last_output = lstm_out[:, -1, :]  # Shape: [batch_size, hidden_size]
        
        # Apply fully connected layer to get final prediction
        out = self.fc(last_output)  # Shape: [batch_size, output_size]
        
        return out


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=10):
    """Train the LSTM model."""
    # Note: train_loader now uses TrialSampler which shuffles trials but keeps sequences within trials ordered
    model = model.to(device)
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batch_count = 0  # Count valid batches
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        # Track NaN statistics
        train_batches_with_nan_inputs = 0
        train_batches_with_nan_targets = 0
        train_batches_with_nan_outputs = 0
        train_batches_with_nan_loss = 0
        
        for batch_data in train_pbar:
            # Unpack the batch data - now includes trial indices
            inputs, targets, _ = batch_data  # Ignore trial indices during training
            
            # Check for NaN values in inputs and targets
            if torch.isnan(inputs).any():
                train_batches_with_nan_inputs += 1
                train_pbar.set_postfix({'status': 'NaN inputs - skipped'})
                continue
                
            if torch.isnan(targets).any():
                train_batches_with_nan_targets += 1
                train_pbar.set_postfix({'status': 'NaN targets - skipped'})
                continue
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Check for NaN in outputs
            if torch.isnan(outputs).any():
                train_batches_with_nan_outputs += 1
                train_pbar.set_postfix({'status': 'NaN outputs - skipped'})
                continue
                
            loss = criterion(outputs, targets)
            
            # Check if loss is NaN
            if torch.isnan(loss):
                train_batches_with_nan_loss += 1
                train_pbar.set_postfix({'status': 'NaN loss - skipped'})
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batch_count += 1
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Print NaN statistics
        if train_batches_with_nan_inputs > 0 or train_batches_with_nan_targets > 0 or train_batches_with_nan_outputs > 0 or train_batches_with_nan_loss > 0:
            print(f"\nTraining NaN statistics for epoch {epoch+1}:")
            print(f"  Batches with NaN inputs: {train_batches_with_nan_inputs}/{len(train_loader)} ({train_batches_with_nan_inputs/len(train_loader)*100:.2f}%)")
            print(f"  Batches with NaN targets: {train_batches_with_nan_targets}/{len(train_loader)} ({train_batches_with_nan_targets/len(train_loader)*100:.2f}%)")
            print(f"  Batches with NaN outputs: {train_batches_with_nan_outputs}/{len(train_loader)} ({train_batches_with_nan_outputs/len(train_loader)*100:.2f}%)")
            print(f"  Batches with NaN loss: {train_batches_with_nan_loss}/{len(train_loader)} ({train_batches_with_nan_loss/len(train_loader)*100:.2f}%)")
        
        # Handle case with no valid batches
        if train_batch_count == 0:
            print(f"\nWarning: No valid training batches in epoch {epoch+1}. Skipping validation.")
            continue
            
        avg_train_loss = train_loss / train_batch_count
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batch_count = 0  # Count valid batches
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        # Track NaN statistics
        val_batches_with_nan_inputs = 0
        val_batches_with_nan_targets = 0
        val_batches_with_nan_outputs = 0
        val_batches_with_nan_loss = 0
        
        with torch.no_grad():
            for batch_data in val_pbar:
                # Unpack the batch data - now includes trial indices
                inputs, targets, _ = batch_data  # Ignore trial indices during validation
                
                # Check for NaN values
                if torch.isnan(inputs).any():
                    val_batches_with_nan_inputs += 1
                    val_pbar.set_postfix({'status': 'NaN inputs - skipped'})
                    continue
                    
                if torch.isnan(targets).any():
                    val_batches_with_nan_targets += 1
                    val_pbar.set_postfix({'status': 'NaN targets - skipped'})
                    continue
                
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # Check for NaN in outputs
                if torch.isnan(outputs).any():
                    val_batches_with_nan_outputs += 1
                    val_pbar.set_postfix({'status': 'NaN outputs - skipped'})
                    continue
                
                loss = criterion(outputs, targets)
                
                # Skip NaN losses
                if torch.isnan(loss):
                    val_batches_with_nan_loss += 1
                    val_pbar.set_postfix({'status': 'NaN loss - skipped'})
                    continue
                    
                val_loss += loss.item()
                val_batch_count += 1
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Print NaN statistics
        if val_batches_with_nan_inputs > 0 or val_batches_with_nan_targets > 0 or val_batches_with_nan_outputs > 0 or val_batches_with_nan_loss > 0:
            print(f"\nValidation NaN statistics for epoch {epoch+1}:")
            print(f"  Batches with NaN inputs: {val_batches_with_nan_inputs}/{len(val_loader)} ({val_batches_with_nan_inputs/len(val_loader)*100:.2f}%)")
            print(f"  Batches with NaN targets: {val_batches_with_nan_targets}/{len(val_loader)} ({val_batches_with_nan_targets/len(val_loader)*100:.2f}%)")
            print(f"  Batches with NaN outputs: {val_batches_with_nan_outputs}/{len(val_loader)} ({val_batches_with_nan_outputs/len(val_loader)*100:.2f}%)")
            print(f"  Batches with NaN loss: {val_batches_with_nan_loss}/{len(val_loader)} ({val_batches_with_nan_loss/len(val_loader)*100:.2f}%)")
        
        # Handle case with no valid batches
        if val_batch_count == 0:
            print(f"\nWarning: No valid validation batches in epoch {epoch+1}. Skipping evaluation.")
            continue
            
        avg_val_loss = val_loss / val_batch_count
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {avg_train_loss:.4f} (from {train_batch_count}/{len(train_loader)} batches)")
        print(f"  Val Loss: {avg_val_loss:.4f} (from {val_batch_count}/{len(val_loader)} batches)")
        
        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            patience_counter = 0
            print(f"  New best model! Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    return best_model, best_val_loss

def evaluate_model(model, test_loader, criterion, device, output_features, y_scaler, output_dir, trial_splits, filtered_to_original):
    """Evaluate the model on test data and save predictions as .npz files."""
    model.eval()
    all_targets = []
    all_predictions = []
    all_losses = []
    trial_indices = []  # To store original trial indices for each batch
    
    # Debug: Print information about test dataset
    print(f"\nDebug - Test dataset info:")
    print(f"Number of test trials: {len(trial_splits['test'])}")
    print(f"Test trial indices: {sorted(trial_splits['test'])}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    print(f"Number of batches: {len(test_loader)}")
    
    # Debug: Track unique trial indices
    unique_trial_indices = set()
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Evaluating")):
            try:
                # Unpack the batch data - now includes trial indices
                inputs, targets, batch_trial_indices = batch_data
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Check for NaN values in inputs
                if torch.isnan(inputs).any():
                    print(f"Warning: NaN values found in inputs batch {batch_idx}. Skipping batch.")
                    continue
                
                # Check for NaN values in targets
                if torch.isnan(targets).any():
                    print(f"Warning: NaN values found in targets batch {batch_idx}. Skipping batch.")
                    continue
                
                outputs = model(inputs)
                
                # Check for NaN values in model outputs
                if torch.isnan(outputs).any():
                    print(f"Warning: NaN values found in model outputs for batch {batch_idx}. Skipping batch.")
                    continue
                
                loss = criterion(outputs, targets)
                
                all_targets.append(targets.cpu().numpy())
                all_predictions.append(outputs.cpu().numpy())
                all_losses.append(loss.item())
                
                # Add trial indices from the batch
                trial_indices.extend(batch_trial_indices.cpu().numpy().tolist())
                
                # Update unique trial indices
                unique_trial_indices.update(batch_trial_indices.cpu().numpy().tolist())
                
                # Debug: Print batch trial indices (first and last batch only)
                if batch_idx == 0 or batch_idx == len(test_loader) - 1:
                    print(f"\nDebug - Batch {batch_idx} trial indices:")
                    print(f"Batch size: {len(batch_trial_indices)}")
                    print(f"Unique trial indices in batch: {sorted(set(batch_trial_indices.cpu().numpy().tolist()))}")
                    print(f"Trial indices count: {len(batch_trial_indices)}")
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                traceback.print_exc()  # Print full traceback for debugging
                continue
    
    # Debug: Print summary of trial indices
    print(f"\nDebug - Trial indices summary:")
    print(f"Total trial indices collected: {len(trial_indices)}")
    print(f"Unique trial indices: {sorted(unique_trial_indices)}")
    print(f"Number of unique trial indices: {len(unique_trial_indices)}")
    
    # Check if we have any predictions
    if not all_predictions:
        print("Error: No predictions were generated. Check your test data and model.")
        return {}
    
    # Concatenate results
    all_targets = np.vstack(all_targets)
    all_predictions = np.vstack(all_predictions)
    
    # Debug: Check shapes
    print(f"\nDebug - Shapes:")
    print(f"all_targets shape: {all_targets.shape}")
    print(f"all_predictions shape: {all_predictions.shape}")
    print(f"trial_indices length: {len(trial_indices)}")
    
    # Check for NaN values after stacking
    nan_in_targets = np.isnan(all_targets).any()
    nan_in_predictions = np.isnan(all_predictions).any()
    
    if nan_in_targets or nan_in_predictions:
        print("\nWARNING: NaN values detected after stacking:")
        print(f"  NaN values in targets: {np.isnan(all_targets).sum()}/{all_targets.size} ({np.isnan(all_targets).sum()/all_targets.size*100:.2f}%)")
        print(f"  NaN values in predictions: {np.isnan(all_predictions).sum()}/{all_predictions.size} ({np.isnan(all_predictions).sum()/all_predictions.size*100:.2f}%)")
    
    # Inverse transform to get original scale
    if y_scaler:
        try:
            all_targets = y_scaler.inverse_transform(all_targets)
            all_predictions = y_scaler.inverse_transform(all_predictions)
            
            # Check for NaN values after inverse transform
            nan_in_targets_after = np.isnan(all_targets).any()
            nan_in_predictions_after = np.isnan(all_predictions).any()
            
            if nan_in_targets_after or nan_in_predictions_after:
                print("\nWARNING: NaN values detected after inverse transform:")
                print(f"  NaN values in targets: {np.isnan(all_targets).sum()}/{all_targets.size} ({np.isnan(all_targets).sum()/all_targets.size*100:.2f}%)")
                print(f"  NaN values in predictions: {np.isnan(all_predictions).sum()}/{all_predictions.size} ({np.isnan(all_predictions).sum()/all_predictions.size*100:.2f}%)")
        except Exception as e:
            print(f"Error during inverse transform: {str(e)}")
            traceback.print_exc()  # Print full traceback for debugging
    
    # Save predictions and targets only (without trial indices)
    npz_path = output_dir / "predictions.npz"
    np.savez(npz_path, targets=all_targets, predictions=all_predictions)
    print(f"Predictions saved to {npz_path} (without trial indices)")
    
    # Save the correct unique trial indices to a separate file (renamed from debug_trial_indices.npz)
    indices_path = output_dir / "trial_indices.npz"
    np.savez(indices_path, unique_trial_indices=sorted(unique_trial_indices))
    print(f"Unique trial indices saved to {indices_path}")
    
    # Calculate metrics
    metrics = {}
    for i, feature in enumerate(output_features):
        target_values = all_targets[:, i]
        pred_values = all_predictions[:, i]
        
        # Check for NaN values
        if np.isnan(target_values).any() or np.isnan(pred_values).any():
            print(f"\nWARNING: NaN values detected for feature {feature}. Skipping metrics calculation.")
            
            # Count NaN values
            nan_targets = np.isnan(target_values).sum()
            nan_preds = np.isnan(pred_values).sum()
            print(f"  NaN values in targets: {nan_targets}/{len(target_values)} ({nan_targets/len(target_values)*100:.2f}%)")
            print(f"  NaN values in predictions: {nan_preds}/{len(pred_values)} ({nan_preds/len(pred_values)*100:.2f}%)")
            
            # Store NaN metrics to indicate issue
            metrics[feature] = {
                'mae': np.nan,
                'mse': np.nan,
                'rmse': np.nan,
                'r2': np.nan,
                'nan_targets': nan_targets,
                'nan_predictions': nan_preds
            }
            continue
        
        try:
            # Calculate metrics with valid data
            mae = mean_absolute_error(target_values, pred_values)
            mse = mean_squared_error(target_values, pred_values)
            rmse = np.sqrt(mse)
            r2 = r2_score(target_values, pred_values)
            
            metrics[feature] = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }
            
            print(f"\nMetrics for {feature}:")
            print(f"  MAE: {mae:.4f}")
            print(f"  MSE: {mse:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²: {r2:.4f}")
        except Exception as e:
            print(f"\nError calculating metrics for feature {feature}: {str(e)}")
            metrics[feature] = {
                'error': str(e),
                'mae': np.nan,
                'mse': np.nan,
                'rmse': np.nan,
                'r2': np.nan
            }
    
    return metrics

def objective(trial, data_loaders, input_features, output_features, device, leg_prefix):
    """Optuna objective function for hyperparameter optimization."""
    # Note: train_loader in data_loaders[0] uses TrialSampler which shuffles trials but keeps sequences within trials ordered
    
    # Suggest hyperparameters
    config = {
        'hidden_size': trial.suggest_int('hidden_size', 32, 256),
        'num_layers': trial.suggest_int('num_layers', 1, 4),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    }
    
    # Create model with suggested hyperparameters
    model = LSTMPredictor(
        input_size=len(input_features),
        hidden_size=config['hidden_size'],
        output_size=len(output_features),
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    # Use DerivativeLoss
    criterion = DerivativeLoss(alpha=0.5)
    print(f"\nUsing DerivativeLoss with alpha={criterion.alpha}")
    print(f"  - This combines standard MAE loss ({1-criterion.alpha:.2f}) with derivative matching ({criterion.alpha:.2f})")
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    train_loader, val_loader = data_loaders
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(50):  # Maximum epochs for hyperparameter search
        # Training phase
        model.train()
        for batch_data in train_loader:
            # Unpack the batch data - now includes trial indices
            inputs, targets, _ = batch_data  # Ignore trial indices during training
            
            # Skip batches with NaN values
            if torch.isnan(inputs).any() or torch.isnan(targets).any():
                continue
                
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Skip if outputs contain NaN
            if torch.isnan(outputs).any():
                continue
                
            loss = criterion(outputs, targets)
            
            # Skip NaN losses
            if torch.isnan(loss):
                continue
                
            loss.backward()
            optimizer.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_count = 0  # Keep track of valid batches
        
        with torch.no_grad():
            for batch_data in val_loader:
                # Unpack the batch data - now includes trial indices
                inputs, targets, _ = batch_data  # Ignore trial indices during validation
                
                # Skip batches with NaN values
                if torch.isnan(inputs).any() or torch.isnan(targets).any():
                    continue
                    
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # Skip if outputs contain NaN
                if torch.isnan(outputs).any():
                    continue
                    
                loss = criterion(outputs, targets)
                
                # Skip NaN losses
                if torch.isnan(loss):
                    continue
                    
                val_loss += loss.item()
                val_count += 1
        
        # Avoid division by zero
        if val_count == 0:
            print("Warning: No valid validation batches found. Returning maximum loss.")
            return float('inf')
            
        val_loss = val_loss / val_count
        
        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 1:  # Early stopping patience
                break
        
        # Report intermediate value
        trial.report(val_loss, epoch)
        
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return best_val_loss

def save_trial_splits_info(trial_splits, filtered_to_original, filtered_trials, output_dir, leg_name):
    """Save information about trial splits to text files."""
    # Create directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save trial splits
    splits_file = output_dir / f"{leg_name}_trial_splits.txt"
    with open(splits_file, 'w') as f:
        f.write(f"Trial splits information for {leg_name}\n")
        f.write(f"Generated at: {datetime.now()}\n\n")
        
        for split_name, indices in trial_splits.items():
            f.write(f"{split_name.upper()} SPLIT:\n")
            f.write(f"Number of trials: {len(indices)}\n")
            f.write(f"Filtered indices: {sorted(indices)}\n")
            f.write(f"Original indices: {sorted(filtered_to_original[idx] for idx in indices)}\n\n")
    
    # Save mapping between filtered and original indices
    mapping_file = output_dir / f"{leg_name}_index_mapping.txt"
    with open(mapping_file, 'w') as f:
        f.write(f"Mapping between filtered and original trial indices for {leg_name}\n")
        f.write(f"Generated at: {datetime.now()}\n\n")
        
        f.write("Filtered Index | Original Index\n")
        f.write("------------------------------\n")
        
        for filtered_idx, original_idx in filtered_to_original.items():
            f.write(f"{filtered_idx:13} | {original_idx}\n")
    
    # Save detailed trial information
    trials_info_file = output_dir / f"{leg_name}_trial_info.txt"
    with open(trials_info_file, 'w') as f:
        f.write(f"Detailed trial information for {leg_name}\n")
        f.write(f"Generated at: {datetime.now()}\n\n")
        
        for filtered_idx, trial_data in enumerate(filtered_trials):
            original_idx = filtered_to_original[filtered_idx]
            
            genotype = trial_data['genotype'].iloc[0] if 'genotype' in trial_data.columns else 'Unknown'
            
            # Calculate average velocities
            if 'x_vel' in trial_data.columns:
                avg_x_vel = trial_data['x_vel'].mean()
                avg_y_vel = trial_data['y_vel'].mean()
                avg_z_vel = trial_data['z_vel'].mean()
                
                # Determine which split this trial belongs to
                split = None
                for split_name, indices in trial_splits.items():
                    if filtered_idx in indices:
                        split = split_name
                        break
                
                f.write(f"Trial {filtered_idx} (Original index: {original_idx})\n")
                f.write(f"  Genotype: {genotype}\n")
                f.write(f"  Split: {split}\n")
                f.write(f"  Average velocities:\n")
                f.write(f"    x: {avg_x_vel:.2f} mm/s\n")
                f.write(f"    y: {avg_y_vel:.2f} mm/s\n")
                f.write(f"    z: {avg_z_vel:.2f} mm/s\n")
                f.write("\n")
    
    return splits_file, mapping_file, trials_info_file

def main():
    """Main function to train and evaluate LSTM models for all 6 legs."""
    print("\nTraining Strategy: Shuffling trials but keeping sequences within trials in order.")
    print("This maintains temporal relationships within each trial while randomizing between trials.")
    
    # Configuration
    base_config = {
        'sequence_length': 50,
        'hidden_size': 256,
        'num_layers': 2,
        'dropout': 0.2,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 200,
        'patience': 10
    }
    
    # Create base directories
    results_dir = Path('lstm_results')
    results_dir.mkdir(exist_ok=True, parents=True)
    print("\nCreated base results directory:", results_dir)
    
    # Create error log file
    error_log = results_dir / 'error_log.txt'
    with open(error_log, 'w') as f:
        f.write(f"Training started at {datetime.now()}\n")
        f.write("-" * 80 + "\n\n")
    
    # Define legs and their features
    legs = {
        'R1': {
            'angles': ['R1A_flex', 'R1A_rot', 'R1A_abduct', 'R1B_flex', 'R1B_rot', 'R1C_flex', 'R1C_rot', 'R1D_flex']
        },
        'L1': {
            'angles': ['L1A_flex', 'L1A_rot', 'L1A_abduct', 'L1B_flex', 'L1B_rot', 'L1C_flex', 'L1C_rot', 'L1D_flex']
        },
        'R2': {
            'angles': ['R2A_flex', 'R2A_rot', 'R2A_abduct', 'R2B_flex', 'R2B_rot', 'R2C_flex', 'R2C_rot', 'R2D_flex']
        },
        'L2': {
            'angles': ['L2A_flex', 'L2A_rot', 'L2A_abduct', 'L2B_flex', 'L2B_rot', 'L2C_flex', 'L2C_rot', 'L2D_flex']
        },
        'R3': {
            'angles': ['R3A_flex', 'R3A_rot', 'R3A_abduct', 'R3B_flex', 'R3B_rot', 'R3C_flex', 'R3C_rot', 'R3D_flex']
        },
        'L3': {
            'angles': ['L3A_flex', 'L3A_rot', 'L3A_abduct', 'L3B_flex', 'L3B_rot', 'L3C_flex', 'L3C_rot', 'L3D_flex']
        }
    }
    
    # Get the data path using any leg name (all should point to the same file)
    data_path = get_available_data_path(next(iter(legs.keys())))
    print(f"\nUsing data path: {data_path}")
    
    # Load data once to create trial splits - using first leg as reference
    print("\nLoading data once to generate consistent trial splits for all legs...")
    first_leg = next(iter(legs.keys()))
    print(f"Using {first_leg} as reference leg for trial split generation")
    
    # Load and prepare data once just to get the trial splits
    # Note: These are temporary loaders and scalers that we'll discard
    (
        _, _, _, _, _,
        _, _, _,
        _, _, shared_trial_splits, shared_filtered_to_original
    ) = prepare_data(
        data_path=data_path,
        input_features=['x_vel', 'y_vel', 'z_vel'],  # Minimal features just to get the splits
        output_features=legs[first_leg]['angles'],
        sequence_length=base_config['sequence_length'],
        output_dir=results_dir,
        leg_name="trial_splits_reference"
    )
    
    # Save master trial splits to base directory
    master_splits_file = results_dir / "master_trial_splits.json"
    with open(master_splits_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_splits = {
            split: sorted(indices) 
            for split, indices in shared_trial_splits.items()
        }
        json.dump(serializable_splits, f, indent=2)
    
    # Save master filtered to original mapping
    master_mapping_file = results_dir / "master_filtered_to_original.json"
    with open(master_mapping_file, 'w') as f:
        # Convert keys to strings for JSON serialization
        serializable_mapping = {
            str(filtered_idx): original_idx 
            for filtered_idx, original_idx in shared_filtered_to_original.items()
        }
        json.dump(serializable_mapping, f, indent=2)
    
    print(f"\nMaster trial splits saved to: {master_splits_file}")
    print(f"Master filtered-to-original mapping saved to: {master_mapping_file}")
    print("\nConsistent trial splits across legs:")
    for split_name, indices in shared_trial_splits.items():
        print(f"  {split_name.capitalize()}: {len(indices)} trials - {sorted(indices)[:10]}...")
    
    # Train model for each leg using the same trial splits
    for leg_name, leg_info in legs.items():
        try:
            print(f"\n{'='*80}")
            print(f"Training model for {leg_name} leg")
            print(f"{'='*80}")
            
            # Create leg-specific directories
            leg_dir = Path(f"lstm_results/{leg_name}")
            leg_models_dir = leg_dir / "models"
            leg_plots_dir = leg_dir / "plots"
            
            os.makedirs(leg_dir, exist_ok=True)
            os.makedirs(leg_models_dir, exist_ok=True)
            os.makedirs(leg_plots_dir, exist_ok=True)
            
            print(f"\nCreated directory structure for {leg_name}:")
            print(f"- {leg_dir}")
            print(f"- {leg_models_dir}")
            print(f"- {leg_plots_dir}")
            
            # Get data path - same for all legs
            data_path = get_available_data_path(leg_name)
            
            # Use leg-specific output features
            output_features = leg_info['angles']
            
            # Define input features
            input_features = [
                'x_vel_ma5',    # Moving average velocities
                'y_vel_ma5',
                'z_vel_ma5',
                'x_vel_ma10',
                'y_vel_ma10',
                'z_vel_ma10',
                'x_vel_ma20',
                'y_vel_ma20',
                'z_vel_ma20',
                'x_vel',        # Raw velocities
                'y_vel',
                'z_vel',
                'x_acc_ma5',    # Acceleration
                'y_acc_ma5',
                'z_acc_ma5',
                'x_acc_ma10',
                'y_acc_ma10',
                'z_acc_ma10',
                'x_acc_ma20',
                'y_acc_ma20',
                'z_acc_ma20',
                'x_acc',        # Raw acceleration
                'y_acc',
                'z_acc',
                'vel_mag_ma5',  # Velocity magnitude
                'vel_mag_ma10',
                'vel_mag_ma20',
                'vel_mag',      # Raw velocity magnitude
                'acc_mag_ma5',  # Acceleration magnitude
                'acc_mag_ma10',
                'acc_mag_ma20',
                'acc_mag',      # Raw acceleration magnitude
                'x_vel_lag1',   # Lagged features
                'y_vel_lag1',
                'z_vel_lag1',
                'x_vel_lag2',
                'y_vel_lag2',
                'z_vel_lag2',
                'x_vel_lag3',
                'y_vel_lag3',
                'z_vel_lag3',
                'x_vel_lag4',
                'y_vel_lag4',
                'z_vel_lag4',
                'x_vel_lag5',
                'y_vel_lag5',
                'z_vel_lag5',
                'x_vel_psd',    # Power Spectral Density features
                'y_vel_psd',
                'z_vel_psd'
            ]
            
            print(f"\nFeatures for {leg_name}:")
            print(f"Input features ({len(input_features)}):")
            for feat in input_features:
                print(f"  - {feat}")
            print(f"\nOutput features ({len(output_features)}):")
            for feat in output_features:
                print(f"  - {feat}")
            
            print(f"\nUsing consistent trial splits across all legs:")
            print(f"  Train: {len(shared_trial_splits['train'])} trials")
            print(f"  Validation: {len(shared_trial_splits['val'])} trials")
            print(f"  Test: {len(shared_trial_splits['test'])} trials")
            
            # Prepare data for this leg - reusing the shared trial splits
            (
                df, all_filtered_trials, train_trials, val_trials, test_trials,
                train_loader, val_loader, test_loader,
                X_scaler, y_scaler, trial_splits, filtered_to_original
            ) = prepare_data(
                data_path=data_path,
                input_features=input_features,
                output_features=output_features,
                sequence_length=base_config['sequence_length'],
                output_dir=leg_dir,  # Pass the leg directory
                leg_name=leg_name,   # Pass the leg name
                fixed_trial_splits=shared_trial_splits,  # Pass the shared trial splits
                fixed_filtered_to_original=shared_filtered_to_original  # Pass the shared mapping
            )
            
            # Verify we're using the same trial splits
            for split in ['train', 'val', 'test']:
                if set(trial_splits[split]) != set(shared_trial_splits[split]):
                    print(f"WARNING: {leg_name} {split} split doesn't match the master splits!")
                    print(f"  Master: {sorted(shared_trial_splits[split])}")
                    print(f"  {leg_name}: {sorted(trial_splits[split])}")
            
            # Save trial splits to text file
            splits_file, mapping_file, trials_info_file = save_trial_splits_info(trial_splits, filtered_to_original, all_filtered_trials, leg_dir, leg_name)
            
            print(f"\nTrial splits saved to: {splits_file}")
            print(f"Index mapping saved to: {mapping_file}")
            print(f"Detailed trial info saved to: {trials_info_file}")
            
            # Check for NaN values in the dataset
            print("\nChecking for NaN values in processed data...")
            
            # Sample a few batches from each loader to check for NaNs
            for loader_name, loader in [("train", train_loader), ("validation", val_loader), ("test", test_loader)]:
                nan_in_inputs = 0
                nan_in_targets = 0
                
                for i, batch_data in enumerate(loader):
                    if i >= 5:  # Check first 5 batches
                        break
                    
                    # Unpack the batch data - now includes trial indices
                    inputs, targets, _ = batch_data  # Ignore trial indices for NaN check
                        
                    if torch.isnan(inputs).any():
                        nan_in_inputs += 1
                    
                    if torch.isnan(targets).any():
                        nan_in_targets += 1
                
                if nan_in_inputs > 0 or nan_in_targets > 0:
                    print(f"Warning: Found NaN values in {loader_name} loader:")
                    print(f"  Batches with NaN inputs: {nan_in_inputs}/5")
                    print(f"  Batches with NaN targets: {nan_in_targets}/5")
            
            # Create model for this leg
            model = LSTMPredictor(
                input_size=len(input_features),
                hidden_size=base_config['hidden_size'],
                output_size=len(output_features),
                num_layers=base_config['num_layers'],
                dropout=base_config['dropout']
            )
            
            # Set up training
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Use DerivativeLoss instead of L1Loss
            criterion = DerivativeLoss(alpha=0.5)
            print(f"\nUsing DerivativeLoss with alpha={criterion.alpha}")
            
            optimizer = optim.Adam(model.parameters(), lr=base_config['learning_rate'])
            
            # Train model
            best_model_state, best_val_loss = train_model(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                base_config['num_epochs'],
                device,
                base_config['patience']
            )
            
            print(f"\nTraining completed for {leg_name}. Best validation loss: {best_val_loss:.4f}")
            
            # Load best model
            model.load_state_dict(best_model_state)
            
            # Evaluate model and save predictions
            metrics = evaluate_model(
                model,
                test_loader,
                criterion,
                device,
                output_features,
                y_scaler,
                leg_plots_dir,
                trial_splits,
                filtered_to_original
            )
            
            # Save model and results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_save_path = leg_models_dir / f'{leg_name}_lstm_{timestamp}.pth'
            
            torch.save({
                'model_state_dict': best_model_state,
                'config': base_config,
                'input_features': input_features,
                'output_features': output_features,
                'X_scaler': X_scaler,
                'y_scaler': y_scaler,
                'metrics': metrics,
                'best_val_loss': best_val_loss,
                'loss_type': 'DerivativeLoss',
                'loss_params': {'alpha': criterion.alpha},
                'trial_splits': trial_splits,
                'filtered_to_original': filtered_to_original
            }, model_save_path)
            
            print(f"\nModel and results saved to: {model_save_path}")
            
            # Log successful completion
            with open(error_log, 'a') as f:
                f.write(f"Successfully completed training for {leg_name} at {datetime.now()}\n")
                f.write(f"Best validation loss: {best_val_loss:.4f}\n")
                f.write("-" * 80 + "\n\n")
                
        except Exception as e:
            # Log the error
            with open(error_log, 'a') as f:
                f.write(f"Error occurred while processing {leg_name} at {datetime.now()}\n")
                f.write(f"Error type: {type(e).__name__}\n")
                f.write(f"Error message: {str(e)}\n")
                f.write("Traceback:\n")
                f.write(traceback.format_exc())
                f.write("-" * 80 + "\n\n")
            
            print(f"\nError occurred while processing {leg_name}. See {error_log} for details.")
            print(f"Error: {str(e)}")
            continue  # Continue with next leg
    
    print("\nTraining completed for all legs!")
    print(f"Check {error_log} for any errors that occurred during training.")

if __name__ == "__main__":
    main() 