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
import os

# Import Optuna visualization capabilities
import optuna.visualization
import plotly

# Import prepare_data_no_windows from no_window_data
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

# Add import from no_window_data
from no_window_data import (
    WholeTrialDataset,
    prepare_data_no_windows,
)

class Chomp1d(nn.Module):
    """Helper module to ensure causal convolutions"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    """TCN block with dilated causal convolutions"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """Initialize weights using normal initialization"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for leg angle prediction"""
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.2, sequence_length=50):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        # Store sequence_length as attribute
        self.sequence_length = sequence_length
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=(kernel_size-1) * dilation_size,
                    dropout=dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
        # Final layer to map to output dimensions
        self.final = nn.Linear(num_channels[-1], output_size)
        
        print("\nTCN Architecture:")
        print(f"Input size: {input_size}")
        print(f"Hidden channels: {num_channels}")
        print(f"Kernel size: {kernel_size}")
        print(f"Sequence length: {sequence_length}")
        print(f"Output size: {output_size}")
        print(f"Dropout: {dropout}")
        
    def forward(self, x):
        # Ensure input is in the correct format [batch, features, sequence]
        if x.shape[1] != x.shape[2] and x.shape[1] == self.sequence_length:
            # If input is [batch, sequence, features], transpose to [batch, features, sequence]
            x = x.transpose(1, 2)
            
        # Process through TCN layers
        out = self.network(x)  # Shape: [batch, channels, sequence]
        
        # Apply final layer to each time step
        # First transpose to [batch, sequence, channels]
        out = out.transpose(1, 2)
        
        # Apply final layer to each timestep
        batch_size, seq_len, channels = out.shape
        out = out.reshape(-1, channels)  # Reshape to [batch*seq_len, channels]
        out = self.final(out)  # Shape: [batch*seq_len, output_size]
        
        # Reshape back to [batch, sequence, output_size]
        out = out.reshape(batch_size, seq_len, -1)
        
        return out

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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=10):
    """Train the TCN model with whole-trial data."""
    # Note: Using whole-trial data with batch_size=1
    model = model.to(device)
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Debug: Check for NaN values in the first batch of the training set at the beginning of the epoch
        first_batch = None
        try:
            # Get the first batch from the iterator without disturbing the DataLoader
            for batch in train_loader:
                first_batch = batch
                break
                
            if first_batch is not None:
                inputs, targets, _ = first_batch
                print(f"\nDEBUG - First batch of epoch {epoch+1}:")
                print(f"  Inputs shape: {inputs.shape}")
                print(f"  Targets shape: {targets.shape}")
                print(f"  Inputs has NaN: {torch.isnan(inputs).any().item()}")
                print(f"  Targets has NaN: {torch.isnan(targets).any().item()}")
                
                if torch.isnan(inputs).any():
                    nan_coords = torch.where(torch.isnan(inputs))
                    print(f"  Number of NaN in inputs: {torch.isnan(inputs).sum().item()}/{inputs.numel()}")
                    print(f"  First few NaN coordinates: {list(zip(*[c[:5].tolist() for c in nan_coords]))}")
                
                # Check input ranges
                print(f"  Input range: [{inputs.min().item():.4f}, {inputs.max().item():.4f}]")
                print(f"  Target range: [{targets.min().item():.4f}, {targets.max().item():.4f}]")
                
                # Check if batch can be successfully processed
                inputs_test = inputs.clone().to(device)
                inputs_test = inputs_test.transpose(1, 2)  # Transpose for TCN format
                try:
                    with torch.no_grad():
                        outputs_test = model(inputs_test)
                    print(f"  Model can process the inputs: Yes")
                    print(f"  Outputs shape: {outputs_test.shape}")
                    print(f"  Outputs has NaN: {torch.isnan(outputs_test).any().item()}")
                    print(f"  Outputs range: [{outputs_test.min().item():.4f}, {outputs_test.max().item():.4f}]")
                except Exception as e:
                    print(f"  Model failed to process the inputs: {str(e)}")
        except Exception as e:
            print(f"Error during first batch debug: {str(e)}")
        
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
            
            # For whole-trial processing, inputs shape is [batch=1, trial_length, features]
            # and targets shape is [batch=1, trial_length, outputs]
            
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
            
            # For TCN, we need to transpose inputs to [batch, features, sequence]
            # Model expects [batch, features, sequence] but our data is [batch, sequence, features]
            inputs_transposed = inputs.transpose(1, 2)
            
            # Forward pass to get predictions for the entire sequence
            outputs = model(inputs_transposed)  # Now returns predictions for all timesteps
            
            # Check for NaN in outputs
            if torch.isnan(outputs).any():
                train_batches_with_nan_outputs += 1
                train_pbar.set_postfix({'status': 'NaN outputs - skipped'})
                continue
                
            # Calculate loss across all timesteps (similar to LSTM)
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
                
                # For TCN, we need to transpose inputs to [batch, features, sequence]
                inputs_transposed = inputs.transpose(1, 2)
                
                outputs = model(inputs_transposed)  # Now returns predictions for all timesteps
                
                # Check for NaN in outputs
                if torch.isnan(outputs).any():
                    val_batches_with_nan_outputs += 1
                    val_pbar.set_postfix({'status': 'NaN outputs - skipped'})
                    continue
                
                # Calculate loss across all timesteps (similar to LSTM)
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
        print(f"  Train Loss (Normalized MAE): {avg_train_loss:.4f} (from {train_batch_count}/{len(train_loader)} batches)")
        print(f"  Val Loss (Normalized MAE): {avg_val_loss:.4f} (from {val_batch_count}/{len(val_loader)} batches)")
        
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

def evaluate_model(model, test_loader, criterion, device, output_features, y_scaler, output_dir, trial_splits=None, filtered_to_original=None):
    """Evaluate the model on test data and save predictions as .npz files."""
    model.eval()
    
    # Define the frame range we want to keep (51-651, using zero-based indexing)
    start_frame = 50  # Frame 51 (1-indexed) = index 50 (0-indexed)
    end_frame = 650   # Frame 651 (1-indexed) = index 650 (0-indexed)
    frames_to_keep = end_frame - start_frame + 1  # Should be 601 frames
    
    print(f"\nUsing frame range: {start_frame+1}-{end_frame+1} (1-indexed)")
    print(f"Number of frames to keep per trial: {frames_to_keep}")
    
    # Debug: Print information about test dataset
    print(f"\nDebug - Test dataset info:")
    if trial_splits:
        print(f"Number of test trials: {len(trial_splits['test'])}")
        print(f"Test trial indices: {sorted(trial_splits['test'])}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    print(f"Number of batches: {len(test_loader)}")
    
    # Store predictions and targets
    all_targets_by_trial = []  # Will store sliced targets for each trial
    all_outputs_by_trial = []  # Will store sliced outputs for each trial
    trial_indices = []
    
    # Debug: Track unique trial indices
    unique_trial_indices = set()
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Evaluating")):
            try:
                # Unpack the batch
                inputs, targets, batch_trial_indices = batch_data
                
                # Move to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Check for NaN values in inputs
                if torch.isnan(inputs).any():
                    print(f"Warning: NaN values found in inputs batch {batch_idx}. Skipping batch.")
                    continue
                
                # Check for NaN values in targets
                if torch.isnan(targets).any():
                    print(f"Warning: NaN values found in targets batch {batch_idx}. Skipping batch.")
                    continue
                
                # Forward pass
                outputs = model(inputs)
                
                # Check for NaN values in outputs
                if torch.isnan(outputs).any():
                    print(f"Warning: NaN values found in outputs batch {batch_idx}. Skipping batch.")
                    continue
                
                # Process each trial in the batch
                for trial_idx in range(len(inputs)):
                    # Get trial data
                    trial_targets = targets[trial_idx].cpu().numpy()  # Shape (seq_len, n_features)
                    trial_outputs = outputs[trial_idx].cpu().numpy()  # Shape (seq_len, n_features)
                    
                    # Keep only the specified frames (51-651)
                    if len(trial_targets) >= frames_to_keep:
                        # Slice to keep only frames 51-651
                        sliced_targets = trial_targets[start_frame:end_frame+1]
                        sliced_outputs = trial_outputs[start_frame:end_frame+1]
                        
                        # Store the sliced data
                        all_targets_by_trial.append(sliced_targets)
                        all_outputs_by_trial.append(sliced_outputs)
                        
                        # Store the trial index
                        current_trial_idx = batch_trial_indices[trial_idx].item()
                        trial_indices.extend([current_trial_idx] * frames_to_keep)
                        unique_trial_indices.add(current_trial_idx)
                    else:
                        print(f"Warning: Trial {batch_trial_indices[trial_idx].item()} has only {len(trial_targets)} frames, which is less than the required {frames_to_keep}. Skipping trial.")
                
                # Debug: Print batch trial indices (first and last batch only)
                if batch_idx == 0 or batch_idx == len(test_loader) - 1:
                    print(f"\nDebug - Batch {batch_idx} trial indices:")
                    print(f"Batch size: {len(batch_trial_indices)}")
                    print(f"Unique trial indices in batch: {sorted(set(batch_trial_indices.cpu().numpy().tolist()))}")
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                traceback.print_exc()
                continue
    
    # Debug: Print summary of trial indices
    print(f"\nDebug - Trial indices summary:")
    print(f"Unique trial indices: {sorted(unique_trial_indices)}")
    print(f"Number of unique trial indices: {len(unique_trial_indices)}")
    print(f"Expected total frames: {len(unique_trial_indices) * frames_to_keep}")
    
    # Check if we have any predictions
    if not all_outputs_by_trial:
        print("Error: No predictions were generated.")
        return {}
    
    # Concatenate all trials' data
    all_targets = np.vstack(all_targets_by_trial)  # Shape (n_trials*frames_to_keep, n_features)
    all_outputs = np.vstack(all_outputs_by_trial)  # Shape (n_trials*frames_to_keep, n_features)
    
    # Debug: Print shapes
    print(f"\nDebug - Shapes after concatenation:")
    print(f"all_targets shape: {all_targets.shape}")
    print(f"all_outputs shape: {all_outputs.shape}")
    print(f"trial_indices length: {len(trial_indices)}")
    
    # Check for NaN values
    nan_in_targets = np.isnan(all_targets).any()
    nan_in_outputs = np.isnan(all_outputs).any()
    
    if nan_in_targets or nan_in_outputs:
        print("\nWARNING: NaN values detected:")
        print(f"  NaN values in targets: {np.isnan(all_targets).sum()}/{all_targets.size} ({np.isnan(all_targets).sum()/all_targets.size*100:.2f}%)")
        print(f"  NaN values in outputs: {np.isnan(all_outputs).sum()}/{all_outputs.size} ({np.isnan(all_outputs).sum()/all_outputs.size*100:.2f}%)")
    
    # Inverse transform if scaler is provided
    if y_scaler:
        try:
            all_targets = y_scaler.inverse_transform(all_targets)
            all_outputs = y_scaler.inverse_transform(all_outputs)
            
            # Check for NaN values after inverse transform
            nan_in_targets_after = np.isnan(all_targets).any()
            nan_in_outputs_after = np.isnan(all_outputs).any()
            
            if nan_in_targets_after or nan_in_outputs_after:
                print("\nWARNING: NaN values detected after inverse transform:")
                print(f"  NaN values in targets: {np.isnan(all_targets).sum()}/{all_targets.size} ({np.isnan(all_targets).sum()/all_targets.size*100:.2f}%)")
                print(f"  NaN values in outputs: {np.isnan(all_outputs).sum()}/{all_outputs.size} ({np.isnan(all_outputs).sum()/all_outputs.size*100:.2f}%)")
        except Exception as e:
            print(f"Error during inverse transform: {str(e)}")
            traceback.print_exc()
    
    # Save as 2D arrays to match LSTM format
    npz_path = output_dir / "predictions.npz"
    np.savez(npz_path, targets=all_targets, predictions=all_outputs)
    print(f"Predictions saved to {npz_path}")
    
    # Save unique trial indices
    indices_path = output_dir / "trial_indices.npz"
    np.savez(indices_path, unique_trial_indices=sorted(unique_trial_indices))
    print(f"Unique trial indices saved to {indices_path}")
    
    # Calculate metrics for each feature
    metrics = {}
    for i, feature in enumerate(output_features):
        # Get target and prediction values for this feature
        target_values = all_targets[:, i]
        pred_values = all_outputs[:, i]
        
        # Check for NaN values
        if np.isnan(target_values).any() or np.isnan(pred_values).any():
            print(f"\nWARNING: NaN values detected for feature {feature}. Skipping metrics calculation.")
            
            # Count NaN values
            nan_targets = np.isnan(target_values).sum()
            nan_preds = np.isnan(pred_values).sum()
            print(f"  NaN values in targets: {nan_targets}/{len(target_values)} ({nan_targets/len(target_values)*100:.2f}%)")
            print(f"  NaN values in predictions: {nan_preds}/{len(pred_values)} ({nan_preds/len(pred_values)*100:.2f}%)")
            
            # Store NaN metrics
            metrics[feature] = {
                'mae': np.nan,
                'mse': np.nan,
                'rmse': np.nan,
                'r2': np.nan,
                'nan_targets': nan_targets,
                'nan_predictions': nan_preds
            }
            continue
        
        # Calculate metrics
        try:
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
            print(f"Error calculating metrics for feature {feature}: {str(e)}")
            metrics[feature] = {
                'error': str(e),
                'mae': np.nan,
                'mse': np.nan,
                'rmse': np.nan,
                'r2': np.nan
            }
    
    return metrics

def objective(trial, data_loaders, input_features, output_features, device, leg_prefix):
    """Optuna objective function for hyperparameter optimization with whole-trial processing."""
    # Note: Using whole-trial data with batch_size=1
    
    # Suggest hyperparameters specifically for TCN with whole-trial processing
    config = {
        'hidden_size': trial.suggest_int('hidden_size', 32, 256, step=32),
        'num_levels': trial.suggest_int('num_levels', 4, 8),  # Minimum 4 levels for adequate receptive field
        'kernel_size': trial.suggest_categorical('kernel_size', [5, 7, 9, 11]),  # Larger kernel sizes for 100-frame context
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
    }
    
    # Create num_channels list - decreasing by a factor of 2 each level
    num_channels = [config['hidden_size']]
    for i in range(1, config['num_levels']):
        num_channels.append(num_channels[-1] // 2)
        if num_channels[-1] < 16:  # Ensure we don't go too small
            num_channels[-1] = 16
            
    # Calculate effective receptive field size
    receptive_field = 1 + (config['kernel_size'] - 1) * sum(2**i for i in range(config['num_levels']))
    print(f"\nTCN configuration: kernel_size={config['kernel_size']}, num_levels={config['num_levels']}")
    print(f"Effective receptive field: {receptive_field} frames")
    
    # Get a sample input from the training loader
    sample_batch = next(iter(data_loaders[0]))
    sample_input = sample_batch[0]  # [batch=1, seq_len, features]
    sequence_length = sample_input.shape[1]  # Get the actual sequence length from data
    
    # Create model with suggested hyperparameters
    model = TemporalConvNet(
        input_size=len(input_features),
        output_size=len(output_features),
        num_channels=num_channels,
        kernel_size=config['kernel_size'],
        dropout=config['dropout'],
        sequence_length=sequence_length
    )
    
    # Use standard MAE loss
    criterion = nn.L1Loss()
    print(f"\nUsing standard MAE loss")
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
            
            # For TCN, we need to transpose inputs to [batch, features, sequence]
            inputs = inputs.transpose(1, 2)
            
            optimizer.zero_grad()
            outputs = model(inputs)  # Now outputs all timesteps
            
            # Skip if outputs contain NaN
            if torch.isnan(outputs).any():
                continue
            
            # Calculate loss on all timesteps (not just the last one)
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
                
                # For TCN, we need to transpose inputs to [batch, features, sequence]
                inputs = inputs.transpose(1, 2)
                
                outputs = model(inputs)  # Now outputs all timesteps
                
                # Skip if outputs contain NaN
                if torch.isnan(outputs).any():
                    continue
                
                # Calculate loss on all timesteps
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
    """Main function to train and evaluate TCN models for all 6 legs."""
    print("\nTraining Strategy: Using whole trials for TCN model without windowing.")
    print("This approach preserves the entire temporal context for each trial.")
    print("TCN architecture is well-suited for processing whole sequences through dilated convolutions.")
    print("For 100-frame context, we'll use larger kernel sizes (5-11) and sufficient network depth (4-8 levels).")
    
    # Configuration
    base_config = {
        'pred_len': 1,           # Prediction length for non-windowed approach
        'batch_size': 1,         # Use batch_size=1 for whole trials
        'num_epochs': 200,
        'patience': 10,
        'n_trials': 20,          # Number of Optuna trials
    }
    
    # Create base directories
    results_dir = Path('tcn_results')
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
    result = prepare_data_no_windows(
        data_path=data_path,
        input_features=['x_vel', 'y_vel', 'z_vel'],  # Minimal features just to get the splits
        output_features=legs[first_leg]['angles'],
        pred_len=base_config['pred_len'],
        output_dir=results_dir,
        leg_name="trial_splits_reference"
    )
    
    # Get the trial splits and mapping - they are at indices 13 and 14 (0-indexed)
    shared_trial_splits = result[13]
    shared_filtered_to_original = result[14]
    
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
            leg_dir = Path(f"tcn_results/{leg_name}")
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
            
            # Prepare data for this leg using no_window_data - reusing the shared trial splits
            result = prepare_data_no_windows(
                data_path=data_path,
                input_features=input_features,
                output_features=output_features,
                pred_len=base_config['pred_len'],
                output_dir=leg_dir,  # Pass the leg directory
                leg_name=leg_name,   # Pass the leg name
                fixed_trial_splits=shared_trial_splits,  # Pass the shared trial splits
                fixed_filtered_to_original=shared_filtered_to_original  # Pass the shared mapping
            )
            
            # Unpack the result
            df, all_filtered_trials = result[0], result[1]
            X_train, y_train = result[2], result[3]
            X_val, y_val = result[4], result[5]
            X_test, y_test = result[6], result[7]
            train_loader, val_loader, test_loader = result[8], result[9], result[10]
            X_scaler, y_scaler = result[11], result[12]
            trial_splits, filtered_to_original = result[13], result[14]
            
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
            
            # Set up device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"\nUsing device: {device}")
            
            # Initialize Optuna study
            study_name = f"{leg_name}_tcn_optimization"
            storage_name = f"sqlite:///{leg_dir}/optuna.db"
            
            print(f"\nStarting hyperparameter optimization for {leg_name} with Optuna")
            print(f"Study name: {study_name}")
            print(f"Number of trials: {base_config['n_trials']}")
            
            # Create a new study or continue from a previous one
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_name,
                load_if_exists=True,
                direction="minimize",
                pruner=optuna.pruners.MedianPruner()
            )
            
            # Define the objective function for hyperparameter optimization
            def optuna_objective(trial):
                # Suggest hyperparameters specifically for TCN with whole-trial processing
                config = {
                    'hidden_size': trial.suggest_int('hidden_size', 32, 256, step=32),
                    'num_levels': trial.suggest_int('num_levels', 4, 8),  # Minimum 4 levels for adequate receptive field
                    'kernel_size': trial.suggest_categorical('kernel_size', [5, 7, 9, 11]),  # Larger kernel sizes for 100-frame context
                    'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.1),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                }
                
                # Create num_channels list - decreasing by a factor of 2 each level
                num_channels = [config['hidden_size']]
                for i in range(1, config['num_levels']):
                    num_channels.append(num_channels[-1] // 2)
                    if num_channels[-1] < 16:  # Ensure we don't go too small
                        num_channels[-1] = 16
                
                # Calculate effective receptive field size
                receptive_field = 1 + (config['kernel_size'] - 1) * sum(2**i for i in range(config['num_levels']))
                print(f"\nTCN configuration: kernel_size={config['kernel_size']}, num_levels={config['num_levels']}")
                print(f"Effective receptive field: {receptive_field} frames (needed for 100-frame context)")
                
                # Create model with suggested hyperparameters
                model = TemporalConvNet(
                    input_size=len(input_features),
                    output_size=len(output_features),
                    num_channels=num_channels,
                    kernel_size=config['kernel_size'],
                    dropout=config['dropout'],
                    sequence_length=X_train.shape[1]  # Use actual sequence length from data
                )
                
                # Use standard MAE loss
                criterion = nn.L1Loss()
                print(f"\nTrial {trial.number}: Using standard MAE loss")
                optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
                
                # Train the model
                best_model_state, best_val_loss = train_model(
                    model,
                    train_loader,
                    val_loader,
                    criterion,
                    optimizer,
                    num_epochs=100,  # Limited epochs for hyperparameter search
                    device=device,
                    patience=5      # Reduced patience for faster trials
                )
                
                # Return the best validation loss
                return best_val_loss
            
            # Run the optimization
            study.optimize(
                optuna_objective,
                n_trials=base_config['n_trials']
            )
            
            # Get the best trial
            best_trial = study.best_trial
            best_params = best_trial.params
            
            print(f"\nBest hyperparameters for {leg_name}:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")
            print(f"Best validation loss: {best_trial.value:.4f}")
            
            # Save hyperparameter optimization results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            optuna_results_file = leg_dir / f"optuna_results_{timestamp}.json"
            
            # Prepare results for JSON serialization
            optuna_results = {
                "best_params": best_params,
                "best_value": best_trial.value,
                "study_name": study_name,
                "timestamp": timestamp,
                "trials": [
                    {
                        "number": t.number,
                        "params": t.params,
                        "value": t.value if t.value is not None else float('nan'),
                        "state": t.state.name
                    }
                    for t in study.trials
                ]
            }
            
            with open(optuna_results_file, 'w') as f:
                json.dump(optuna_results, f, indent=2, default=str)
            
            print(f"\nOptuna results saved to: {optuna_results_file}")
            
            # Generate optimization visualization plots
            try:
                # Create optimization history plot
                fig = optuna.visualization.plot_optimization_history(study)
                fig.write_image(str(leg_plots_dir / f"optuna_history_{timestamp}.png"))
                
                # Create parameter importance plot
                param_imp_fig = optuna.visualization.plot_param_importances(study)
                param_imp_fig.write_image(str(leg_plots_dir / f"optuna_param_importance_{timestamp}.png"))
                
                # Create parallel coordinate plot
                parallel_fig = optuna.visualization.plot_parallel_coordinate(study)
                parallel_fig.write_image(str(leg_plots_dir / f"optuna_parallel_coordinate_{timestamp}.png"))
                
                print(f"\nOptuna visualization plots saved to: {leg_plots_dir}")
            except Exception as e:
                print(f"Warning: Could not generate Optuna visualization plots: {str(e)}")
            
            # Train the best model with full epochs
            print(f"\nTraining the best model for {leg_name} with full epochs...")
            
            # Create num_channels list based on best parameters
            num_channels = [best_params['hidden_size']]
            for i in range(1, best_params['num_levels']):
                num_channels.append(num_channels[-1] // 2)
                if num_channels[-1] < 16:  # Ensure we don't go too small
                    num_channels[-1] = 16
            
            # Create model with best hyperparameters
            best_model = TemporalConvNet(
                input_size=len(input_features),
                output_size=len(output_features),
                num_channels=num_channels,
                kernel_size=best_params['kernel_size'],
                dropout=best_params['dropout'],
                sequence_length=X_train.shape[1]  # Use actual sequence length from data
            )
            
            # Use standard MAE loss
            best_criterion = nn.L1Loss()
            print(f"\nUsing standard MAE loss for final training")
            
            best_optimizer = optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
            
            # Train the best model with full epochs
            best_model_state, best_val_loss = train_model(
                best_model,
                train_loader,
                val_loader,
                best_criterion,
                best_optimizer,
                base_config['num_epochs'],
                device,
                base_config['patience']
            )
            
            print(f"\nFull training completed for {leg_name}. Best validation loss: {best_val_loss:.4f}")
            
            # Load best model
            best_model.load_state_dict(best_model_state)
            
            # Evaluate model and save predictions
            metrics = evaluate_model(
                best_model,
                test_loader,
                best_criterion,
                device,
                output_features,
                y_scaler,
                leg_plots_dir,
                trial_splits,
                filtered_to_original
            )
            
            # Save best model and results
            model_save_path = leg_models_dir / f'{leg_name}_tcn_{timestamp}.pth'
            
            torch.save({
                'model_state_dict': best_model_state,
                'best_params': best_params,
                'input_features': input_features,
                'output_features': output_features,
                'X_scaler': X_scaler,
                'y_scaler': y_scaler,
                'metrics': metrics,
                'best_val_loss': best_val_loss,
                'loss_type': 'MAE',
                'model_type': 'TCN',
                'trial_splits': trial_splits,
                'filtered_to_original': filtered_to_original,
                'optuna_study_name': study_name,
                'optuna_best_trial': best_trial.number
            }, model_save_path)
            
            print(f"\nBest TCN model and results saved to: {model_save_path}")
            
            # Log successful completion
            with open(error_log, 'a') as f:
                f.write(f"Successfully completed training for {leg_name} at {datetime.now()}\n")
                f.write(f"Best validation loss: {best_val_loss:.4f}\n")
                f.write(f"Best hyperparameters: {best_params}\n")
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
            traceback.print_exc()  # Print full traceback for debugging
            continue  # Continue with next leg
    
    print("\nTraining completed for all legs!")
    print(f"Check {error_log} for any errors that occurred during training.")

if __name__ == "__main__":
    main() 