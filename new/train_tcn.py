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
from utils.data import TimeSeriesDataset, filter_frames
import optuna
from optuna.trial import TrialState
import shutil
import pickle
import traceback
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from matplotlib.backends.backend_pdf import PdfPages

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
        
    def forward(self, x):
        # Input shape: [batch, features, sequence]
        out = self.network(x)
        # Take only the last timestep: [batch, channels, 1]
        out = out[:, :, -1]
        # Output shape: [batch, output_size]
        return self.final(out)

def filter_trial_frames(X, y, sequence_length):
    """
    Filter frames 400-1000 for each trial, ensuring proper context for sequences.
    
    Args:
        X: Input features array
        y: Target values array
        sequence_length: Length of input sequences for TCN
    """
    filtered_X = []
    filtered_y = []
    frames_per_trial = 1400  # Total frames per trial
    eval_frames = (400, 1000)  # Frame range to evaluate
    num_trials = len(X) // frames_per_trial
    
    print(f"\nFrame filtering details:")
    print(f"  Total frames per trial: {frames_per_trial}")
    print(f"  Evaluation frame range: {eval_frames}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Number of trials: {num_trials}")
    
    for trial in range(num_trials):
        trial_start = trial * frames_per_trial
        
        # Calculate indices for this trial
        eval_start = trial_start + eval_frames[0]  # Frame 400
        eval_end = trial_start + eval_frames[1]    # Frame 1000
        context_start = eval_start - sequence_length + 1  # Include context for first prediction
        
        # Extract frames including context
        trial_X = X[context_start:eval_end]
        trial_y = y[context_start:eval_end]
        
        filtered_X.append(trial_X)
        filtered_y.append(trial_y)
        
        if trial == 0:  # Print details for first trial
            print(f"\nFirst trial frame details:")
            print(f"  Trial start: {trial_start}")
            print(f"  Context start: {context_start} (frame {eval_frames[0] - sequence_length + 1} of trial)")
            print(f"  Eval start: {eval_start} (frame {eval_frames[0]} of trial)")
            print(f"  Eval end: {eval_end} (frame {eval_frames[1]} of trial)")
            print(f"  Total frames: {len(trial_X)}")
            print(f"  Context frames: {eval_start - context_start}")
            print(f"  Evaluation frames: {eval_end - eval_start}")
    
    X_filtered = np.concatenate(filtered_X)
    y_filtered = np.concatenate(filtered_y)
    
    print(f"\nFiltered data shapes:")
    print(f"  X: {X_filtered.shape}")
    print(f"  y: {y_filtered.shape}")
    
    return X_filtered, y_filtered

class SequenceDataset(Dataset):
    """Dataset for TCN training with sequences"""
    def __init__(self, X, y, sequence_length=50):
        """
        Initialize sequence dataset.
        
        Args:
            X (np.ndarray): Input features array of shape (num_samples, num_features)
            y (np.ndarray): Target values array of shape (num_samples, num_targets)
            sequence_length (int): Length of input sequences
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequence_length = sequence_length
        
        # Calculate valid indices that will result in predictions for frames 400-1000
        frames_per_trial = 600  # Number of frames to predict per trial (1000 - 400)
        self.valid_indices = []
        
        # Each trial in the filtered data starts with (sequence_length-1) context frames
        # followed by the frames we want to predict (400-1000)
        total_sequence = sequence_length - 1 + frames_per_trial  # Context + prediction frames
        num_trials = len(self.X) // total_sequence
        
        for trial in range(num_trials):
            trial_start = trial * total_sequence
            # Add indices that will result in predictions for frames 400-1000
            for i in range(frames_per_trial):
                sequence_start = trial_start + i
                self.valid_indices.append(sequence_start)
    
    def __len__(self):
        """Return the number of sequences in the dataset"""
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """Get a sequence and its target"""
        sequence_start = self.valid_indices[idx]
        sequence_end = sequence_start + self.sequence_length
        
        # Get sequence of input features
        # Shape: (sequence_length, num_features) -> (num_features, sequence_length)
        sequence = self.X[sequence_start:sequence_end].transpose(0, 1)
        
        # Get target (last frame's angles)
        target = self.y[sequence_end - 1]
        
        return sequence, target

class ZScoreScaler:
    """Z-score normalization scaler with feature-wise scaling"""
    def __init__(self, means=None, stds=None, feature_names=None):
        self.means = means if means is not None else None
        self.stds = stds if stds is not None else None
        self.feature_names = feature_names
    
    def fit(self, X):
        """Compute mean and std for each feature"""
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        # Prevent division by zero
        self.stds[self.stds == 0] = 1
        return self
    
    def transform(self, X):
        """Apply z-score normalization"""
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if self.means is None or self.stds is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        # Convert means and stds to numpy arrays if they're dictionaries
        if isinstance(self.means, dict):
            means_array = np.array([self.means[feat] for feat in self.feature_names])
            stds_array = np.array([self.stds[feat] for feat in self.feature_names])
            return (X - means_array) / stds_array
        else:
            return (X - self.means) / self.stds
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """Reverse the z-score normalization"""
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if self.means is None or self.stds is None:
            raise ValueError("Scaler has not been fitted yet.")
        
        # Convert means and stds to numpy arrays if they're dictionaries
        if isinstance(self.means, dict):
            means_array = np.array([self.means[feat] for feat in self.feature_names])
            stds_array = np.array([self.stds[feat] for feat in self.feature_names])
            return X * stds_array + means_array
        else:
            return X * self.stds + self.means

def prepare_data(data_path, input_features, output_features, sequence_length, genotype='ALL'):
    """Prepare sequences of data for TCN"""
    print("\nPreparing data...")
    
    # Load and preprocess data
    df = pd.read_csv(data_path)
    print(f"Loaded data with shape: {df.shape}")
    
    # Filter for specified genotype if not ALL
    if genotype != 'ALL':
        df = df[df['genotype'] == genotype].copy()
        print(f"Filtered for {genotype}: {df.shape}")
    
    # Get trial indices (assuming 1400 frames per trial)
    trial_size = 1400
    num_trials = len(df) // trial_size
    print(f"\nTotal number of trials: {num_trials}")
    
    # Calculate moving averages for velocities within each trial
    base_velocities = ['x_vel', 'y_vel', 'z_vel']
    for window in [5, 10, 20]:
        for vel in base_velocities:
            # Calculate moving average for each trial separately
            ma_values = []
            for trial in range(num_trials):
                start_idx = trial * trial_size
                end_idx = (trial + 1) * trial_size
                trial_data = df[vel].iloc[start_idx:end_idx]
                # Calculate moving average and handle edges
                ma = trial_data.rolling(window=window, center=True, min_periods=1).mean()
                ma_values.extend(ma.tolist())
            df[f'{vel}_ma{window}'] = ma_values
    
    # Extract features and targets
    X = df[input_features].values
    y = df[output_features].values
    
    # Reshape data into trials
    X_trials = X.reshape(num_trials, trial_size, -1)
    y_trials = y.reshape(num_trials, trial_size, -1)
    
    # Create trial indices and shuffle
    trial_indices = np.arange(num_trials)
    np.random.seed(42)  # For reproducible splits
    np.random.shuffle(trial_indices)
    
    # Split into train, validation, and test sets
    train_size = int(0.7 * num_trials)
    val_size = int(0.15 * num_trials)
    test_size = num_trials - train_size - val_size
    
    train_trials = trial_indices[:train_size]
    val_trials = trial_indices[train_size:train_size + val_size]
    test_trials = trial_indices[train_size + val_size:]
    
    print("\nSplit Information:")
    print(f"Training: {len(train_trials)} trials")
    print(f"Validation: {len(val_trials)} trials")
    print(f"Test: {len(test_trials)} trials")
    
    # Get data for each split
    X_train = X_trials[train_trials].reshape(-1, X_trials.shape[-1])
    y_train = y_trials[train_trials].reshape(-1, y_trials.shape[-1])
    X_val = X_trials[val_trials].reshape(-1, X_trials.shape[-1])
    y_val = y_trials[val_trials].reshape(-1, y_trials.shape[-1])
    X_test = X_trials[test_trials].reshape(-1, X_trials.shape[-1])
    y_test = y_trials[test_trials].reshape(-1, y_trials.shape[-1])
    
    # Scale input features
    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_val_scaled = X_scaler.transform(X_val)
    X_test_scaled = X_scaler.transform(X_test)
    
    # Calculate z-score parameters for each target independently
    y_means = {}
    y_stds = {}
    y_scaled = np.zeros_like(y_train)
    
    print("\nTarget Statistics and Z-score Verification:")
    for i, feature in enumerate(output_features):
        # Calculate parameters using only training data
        y_means[feature] = np.mean(y_train[:, i])
        y_stds[feature] = np.std(y_train[:, i])
        
        # Z-score all splits
        y_scaled[:, i] = (y_train[:, i] - y_means[feature]) / y_stds[feature]
        
        # Verify z-scoring
        z_mean = np.mean(y_scaled[:, i])
        z_std = np.std(y_scaled[:, i])
        
        print(f"\n{feature}:")
        print(f"  Original - Mean: {y_means[feature]:.2f}°, Std: {y_stds[feature]:.2f}°")
        print(f"  Z-scored - Mean: {z_mean:.6f}, Std: {z_std:.6f}")
    
    # Create scaler for targets
    y_scaler = ZScoreScaler(y_means, y_stds, output_features)
    
    # Scale targets
    y_train_scaled = y_scaler.transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)
    y_test_scaled = y_scaler.transform(y_test)
    
    print("\nFinal Dataset Sizes:")
    print(f"Training: {len(X_train_scaled)} frames")
    print(f"Validation: {len(X_val_scaled)} frames")
    print(f"Test: {len(X_test_scaled)} frames")
    
    # Filter frames 400-1000 for each trial
    X_train_filtered, y_train_filtered = filter_trial_frames(X_train_scaled, y_train_scaled, sequence_length)
    X_val_filtered, y_val_filtered = filter_trial_frames(X_val_scaled, y_val_scaled, sequence_length)
    X_test_filtered, y_test_filtered = filter_trial_frames(X_test_scaled, y_test_scaled, sequence_length)
    
    print("\nDataset sizes after filtering frames 400-1000:")
    print(f"Training: {len(X_train_filtered)} frames")
    print(f"Validation: {len(X_val_filtered)} frames")
    print(f"Test: {len(X_test_filtered)} frames")
    
    # Create sequence datasets
    train_dataset = SequenceDataset(X_train_filtered, y_train_filtered, sequence_length)
    val_dataset = SequenceDataset(X_val_filtered, y_val_filtered, sequence_length)
    test_dataset = SequenceDataset(X_test_filtered, y_test_filtered, sequence_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    return (X_train_filtered, y_train_filtered), (X_val_filtered, y_val_filtered), (X_test_filtered, y_test_filtered), X_scaler, y_scaler

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=20):
    """Train the TCN model"""
    model = model.to(device)  # Move model to device
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    print("\nTraining Configuration:")
    print(f"Device: {device}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")
    print(f"Early stopping patience: {patience}")
    
    # Clear GPU cache if using CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\nGPU Memory after clearing cache:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print("Training phase:")
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # Move data to the same device as model
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'batch': f'{batch_idx+1}/{len(train_loader)}',
                'loss': f'{loss.item():.4f}'
            })
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        print("Validation phase:")
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                # Move data to the same device as model
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'batch': f'{batch_idx+1}/{len(val_loader)}',
                    'loss': f'{loss.item():.4f}'
                })
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Move model state to CPU before saving
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Calculate epoch time
        epoch_time = datetime.now() - epoch_start_time
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Time: {epoch_time}")
        print(f"  Training Loss: {avg_train_loss:.6f}")
        print(f"  Validation Loss: {avg_val_loss:.6f}")
        print(f"  Best Validation Loss: {best_val_loss:.6f}")
        print(f"  Early Stopping Counter: {patience_counter}/{patience}")
        
        if torch.cuda.is_available():
            print(f"  GPU Memory:")
            print(f"    Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f"    Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best validation loss: {best_val_loss:.6f}")
            break
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Total epochs run: {epoch+1}")
    
    return best_model_state, train_losses, val_losses

def evaluate_model(model, test_loader, criterion, device, output_features, output_scaler, save_dir):
    """Evaluate the trained TCN model"""
    # Ensure save directory exists
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()  # Set model to evaluation mode
    
    # Initialize metrics
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Store predictions and targets
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            total_loss += loss.item()
    
    # Calculate average loss
    avg_loss = total_loss / len(test_loader)
    
    # Combine all predictions and targets
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    
    print(f"\nPredictions shape before inverse transform: {predictions.shape}")
    print(f"Targets shape before inverse transform: {targets.shape}")
    
    # Inverse transform predictions and targets
    predictions_original = output_scaler.inverse_transform(predictions)
    targets_original = output_scaler.inverse_transform(targets)
    
    # Calculate frames per trial and number of trials
    frames_per_trial = 600  # 1000 - 400 frames per trial
    num_trials = len(predictions) // frames_per_trial
    total_frames = num_trials * frames_per_trial
    
    print(f"\nReshaping details:")
    print(f"Total predictions: {len(predictions)}")
    print(f"Frames per trial: {frames_per_trial}")
    print(f"Calculated number of trials: {num_trials}")
    print(f"Total frames to use: {total_frames}")
    
    # Trim predictions and targets to fit complete trials
    predictions = predictions[:total_frames]
    targets = targets[:total_frames]
    predictions_original = predictions_original[:total_frames]
    targets_original = targets_original[:total_frames]
    
    # Reshape into trials
    predictions = predictions.reshape(num_trials, frames_per_trial, -1)
    targets = targets.reshape(num_trials, frames_per_trial, -1)
    predictions_original = predictions_original.reshape(num_trials, frames_per_trial, -1)
    targets_original = targets_original.reshape(num_trials, frames_per_trial, -1)
    
    print(f"\nArray shapes after reshaping:")
    print(f"Predictions: {predictions.shape}")
    print(f"Targets: {targets.shape}")
    
    # Calculate metrics for each output feature
    metrics = {
        'test_loss': avg_loss,
        'metrics_by_feature': {}
    }
    
    # Create PDF for prediction plots
    pdf_path = save_dir / 'predictions.pdf'
    with PdfPages(pdf_path) as pdf:
        # For each feature
        for i, feature in enumerate(output_features):
            feature_metrics_z = []
            feature_metrics_raw = []
            
            # Create a figure for this feature with subplots for each trial
            rows = int(np.ceil(num_trials / 2))
            fig, axes = plt.subplots(rows, 2, figsize=(20, 5*rows))
            fig.suptitle(f'{feature} Predictions Across All Trials', fontsize=16)
            axes = axes.flatten()
            
            # Plot each trial
            for trial in range(num_trials):
                # Get predictions and targets for this trial
                trial_pred_z = predictions[trial, :, i]
                trial_target_z = targets[trial, :, i]
                trial_pred_raw = predictions_original[trial, :, i]
                trial_target_raw = targets_original[trial, :, i]
                
                # Calculate metrics for this trial
                mae_z = mean_absolute_error(trial_target_z, trial_pred_z)
                rmse_z = np.sqrt(mean_squared_error(trial_target_z, trial_pred_z))
                r2_z = r2_score(trial_target_z, trial_pred_z)
                
                mae_raw = mean_absolute_error(trial_target_raw, trial_pred_raw)
                rmse_raw = np.sqrt(mean_squared_error(trial_target_raw, trial_pred_raw))
                r2_raw = r2_score(trial_target_raw, trial_pred_raw)
                
                feature_metrics_z.append([mae_z, rmse_z, r2_z])
                feature_metrics_raw.append([mae_raw, rmse_raw, r2_raw])
                
                # Create time axis (frames 400-1000)
                x_axis = np.arange(400, 1000)
                
                # Plot this trial
                ax = axes[trial]
                ax.plot(x_axis, trial_target_raw, 'b-', label='True', alpha=0.7)
                ax.plot(x_axis, trial_pred_raw, 'r-', label='Predicted', alpha=0.7)
                ax.set_title(f'Trial {trial+1}\nMAE: {mae_raw:.2f}°, RMSE: {rmse_raw:.2f}°, R²: {r2_raw:.3f}')
                ax.set_xlabel('Frame')
                ax.set_ylabel('Angle (degrees)')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Remove any empty subplots
            for j in range(trial + 1, len(axes)):
                fig.delaxes(axes[j])
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
            
            # Calculate average metrics across trials
            avg_metrics_z = np.mean(feature_metrics_z, axis=0)
            avg_metrics_raw = np.mean(feature_metrics_raw, axis=0)
            
            metrics['metrics_by_feature'][feature] = {
                'mae_z': float(avg_metrics_z[0]),
                'rmse_z': float(avg_metrics_z[1]),
                'r2_z': float(avg_metrics_z[2]),
                'mae_raw': float(avg_metrics_raw[0]),
                'rmse_raw': float(avg_metrics_raw[1]),
                'r2_raw': float(avg_metrics_raw[2]),
                'trial_metrics': {
                    f'trial_{t+1}': {
                        'mae_z': float(mz[0]),
                        'rmse_z': float(mz[1]),
                        'r2_z': float(mz[2]),
                        'mae_raw': float(mr[0]),
                        'rmse_raw': float(mr[1]),
                        'r2_raw': float(mr[2])
                    }
                    for t, (mz, mr) in enumerate(zip(feature_metrics_z, feature_metrics_raw))
                }
            }
    
    print(f"\nSaved prediction plots to: {pdf_path}")
    
    # Print summary metrics
    print("\nTest Set Metrics (averaged across trials):")
    print(f"Average Loss: {avg_loss:.6f}")
    
    for feature, feature_metrics in metrics['metrics_by_feature'].items():
        print(f"\n{feature}:")
        print(f"  MAE (z-scored): {feature_metrics['mae_z']:.6f}")
        print(f"  RMSE (z-scored): {feature_metrics['rmse_z']:.6f}")
        print(f"  R² (z-scored): {feature_metrics['r2_z']:.6f}")
        print(f"  MAE (raw): {feature_metrics['mae_raw']:.6f}")
        print(f"  RMSE (raw): {feature_metrics['rmse_raw']:.6f}")
        print(f"  R² (raw): {feature_metrics['r2_raw']:.6f}")
        
        print("\n  Trial-wise metrics:")
        for trial_name, trial_metrics in feature_metrics['trial_metrics'].items():
            print(f"    {trial_name}:")
            print(f"      MAE (raw): {trial_metrics['mae_raw']:.6f}°")
            print(f"      RMSE (raw): {trial_metrics['rmse_raw']:.6f}°")
            print(f"      R²: {trial_metrics['r2_raw']:.6f}")
    
    return metrics

def convert_to_serializable(obj):
    """Convert numpy types and Path objects to Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj

def run_experiment(config):
    """Run a TCN experiment with the given configuration"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create model directory
    model_dir = Path('tcn_results') / config['genotype'] / f"{config['leg_prefix']}_leg"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = model_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(convert_to_serializable(config), f, indent=4)
    print(f"\nSaved configuration to: {config_path}")
    
    # Prepare data
    print("\nPreparing data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), X_scaler, y_scaler = prepare_data(
        config['data_path'],
        config['input_features'],
        config['output_features'],
        config['sequence_length'],
        config['genotype']
    )
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = SequenceDataset(X_train, y_train, config['sequence_length'])
    val_dataset = SequenceDataset(X_val, y_val, config['sequence_length'])
    test_dataset = SequenceDataset(X_test, y_test, config['sequence_length'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    print("\nDataset sizes:")
    print(f"Training sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")
    print(f"Test sequences: {len(test_dataset)}")
    
    # Create model
    model = TemporalConvNet(
        input_size=len(config['input_features']),
        output_size=len(config['output_features']),
        num_channels=config['num_channels'],
        kernel_size=config['kernel_size'],
        dropout=config['dropout'],
        sequence_length=config['sequence_length']
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Train model
    print("\nTraining model...")
    best_model_state, train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        config['num_epochs'],
        device,
        patience=config['patience']
    )
    
    # Save model
    model_path = model_dir / 'best_model.pt'
    torch.save({
        'model_state_dict': best_model_state,
        'config': config,
        'X_scaler': X_scaler,
        'y_scaler': y_scaler
    }, model_path)
    print(f"\nSaved best model to: {model_path}")
    
    # Load best model for evaluation
    model.load_state_dict(best_model_state)
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
        config['output_features'],
        y_scaler,
        model_dir
    )
    
    # Save metrics
    metrics_path = model_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(convert_to_serializable(metrics), f, indent=4)
    print(f"\nSaved metrics to: {metrics_path}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_path = model_dir / 'training_curves.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"\nSaved training curves to: {plot_path}")
    
    return metrics

def create_genotype_summary(genotype_dir, experiment_results):
    """Create summary metrics and plots for all legs of a genotype"""
    print("\nCreating genotype summary...")
    
    # Initialize dictionaries for metrics
    leg_metrics = {}
    
    # Process each leg's results
    for leg_prefix, results in experiment_results.items():
        print(f"\nProcessing {leg_prefix} leg metrics...")
        leg_metrics[leg_prefix] = results  # Store metrics directly
    
    # Save summary metrics
    summary_path = Path(genotype_dir) / 'summary_metrics.json'
    with open(summary_path, 'w') as f:
        json.dump(convert_to_serializable(leg_metrics), f, indent=4)
    print(f"\nSaved summary metrics to: {summary_path}")
    
    # Create comparison plots
    create_comparison_plots(leg_metrics, genotype_dir)

def create_comparison_plots(experiment_results, output_dir):
    """Create comparison plots for a leg's experiments"""
    # Sort results by z-scored MAE
    sorted_results_z = []
    sorted_results_raw = []
    
    for leg_prefix, results in experiment_results.items():
        # Calculate average MAE across all features for this leg
        feature_metrics = results['metrics_by_feature']
        avg_mae_z = np.mean([m['mae_z'] for m in feature_metrics.values()])
        avg_mae_raw = np.mean([m['mae_raw'] for m in feature_metrics.values()])
        
        sorted_results_z.append((leg_prefix, avg_mae_z))
        sorted_results_raw.append((leg_prefix, avg_mae_raw))
    
    sorted_results_z.sort(key=lambda x: x[1])
    sorted_results_raw.sort(key=lambda x: x[1])
    
    # Create side-by-side comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Z-scored plot
    legs_z = [r[0] for r in sorted_results_z]
    maes_z = [r[1] for r in sorted_results_z]
    x_positions = np.arange(len(legs_z))
    ax1.bar(x_positions, maes_z)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(legs_z, rotation=45, ha='right')
    ax1.set_title('Average Z-scored MAE by Leg')
    ax1.set_xlabel('Leg')
    ax1.set_ylabel('Z-scored MAE')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on top of each bar
    for i, v in enumerate(maes_z):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Raw plot
    legs_raw = [r[0] for r in sorted_results_raw]
    maes_raw = [r[1] for r in sorted_results_raw]
    x_positions = np.arange(len(legs_raw))
    ax2.bar(x_positions, maes_raw)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(legs_raw, rotation=45, ha='right')
    ax2.set_title('Average Raw MAE by Leg')
    ax2.set_xlabel('Leg')
    ax2.set_ylabel('MAE (degrees)')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on top of each bar
    for i, v in enumerate(maes_raw):
        ax2.text(i, v + 0.01, f'{v:.1f}°', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'leg_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comparison data
    comparison_data = {
        'z_scored_mae': {leg: mae for leg, mae in sorted_results_z},
        'raw_mae': {leg: mae for leg, mae in sorted_results_raw}
    }
    
    with open(output_dir / 'leg_comparison.json', 'w') as f:
        json.dump(comparison_data, f, indent=4)

def main():
    """Main function to run TCN training"""
    # Get base data path
    base_data_path = "Z:/Divya/TEMP_transfers/toAni/BPN_P9LT_P9RT_flyCoords.csv"
    
    # Prompt for genotype selection
    print("\nAvailable genotypes:")
    print("1. BPN")
    print("2. P9LT")
    print("3. P9RT")
    print("4. ALL (train on all genotypes)")
    
    while True:
        try:
            choice = input("\nSelect genotype (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                break
            print("Please enter a number between 1 and 4")
        except ValueError:
            print("Please enter a valid number")
    
    # Map choice to genotype
    genotype_map = {
        '1': 'BPN',
        '2': 'P9LT',
        '3': 'P9RT',
        '4': 'ALL'
    }
    selected_genotype = genotype_map[choice]
    
    print(f"\nSelected genotype: {selected_genotype}")
    
    # Create base results directory for the selected genotype
    base_results_dir = Path('tcn_results') / selected_genotype
    base_results_dir.mkdir(exist_ok=True, parents=True)
    
    # Dictionary to store all experiment results
    all_results = {}
    
    # Define legs and their configurations
    legs = {
        'R-F': {'position': 'R-F-FeTi_z', 'angles': ['R1A_flex', 'R1A_rot', 'R1A_abduct', 'R1B_flex', 'R1B_rot', 'R1C_flex', 'R1C_rot', 'R1D_flex']},
        'L-F': {'position': 'L-F-FeTi_z', 'angles': ['L1A_flex', 'L1A_rot', 'L1A_abduct', 'L1B_flex', 'L1B_rot', 'L1C_flex', 'L1C_rot', 'L1D_flex']},
        'R-M': {'position': 'R-M-FeTi_z', 'angles': ['R2A_flex', 'R2A_rot', 'R2A_abduct', 'R2B_flex', 'R2B_rot', 'R2C_flex', 'R2C_rot', 'R2D_flex']},
        'L-M': {'position': 'L-M-FeTi_z', 'angles': ['L2A_flex', 'L2A_rot', 'L2A_abduct', 'L2B_flex', 'L2B_rot', 'L2C_flex', 'L2C_rot', 'L2D_flex']},
        'R-H': {'position': 'R-H-FeTi_z', 'angles': ['R3A_flex', 'R3A_rot', 'R3A_abduct', 'R3B_flex', 'R3B_rot', 'R3C_flex', 'R3C_rot', 'R3D_flex']},
        'L-H': {'position': 'L-H-FeTi_z', 'angles': ['L3A_flex', 'L3A_rot', 'L3A_abduct', 'L3B_flex', 'L3B_rot', 'L3C_flex', 'L3C_rot', 'L3D_flex']}
    }
    
    # Save configuration
    config_path = base_results_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'genotype': selected_genotype,
            'legs': legs
        }, f, indent=4)
    
    # Process each leg
    for leg_prefix, leg_config in legs.items():
        print(f"\n{'='*80}")
        print(f"Processing {leg_prefix} leg")
        print(f"{'='*80}")
        
        # Create experiment configuration
        config = {
            'data_path': base_data_path,
            'batch_size': 64,
            'learning_rate': 1e-3,
            'num_epochs': 400,
            'patience': 20,
            'sequence_length': 50,
            'kernel_size': 3,
            'dropout': 0.2,
            'num_channels': [64, 128, 256],
            'input_features': [
                'x_vel_ma5', 'y_vel_ma5', 'z_vel_ma5',
                'x_vel_ma10', 'y_vel_ma10', 'z_vel_ma10',
                'x_vel_ma20', 'y_vel_ma20', 'z_vel_ma20',
                'x_vel', 'y_vel', 'z_vel',
             #   leg_config['position']
            ],
            'output_features': leg_config['angles'],
            'genotype': selected_genotype,
            'leg_prefix': leg_prefix
        }
        
        # Run experiment
        metrics = run_experiment(config)
        all_results[leg_prefix] = metrics
    
    # Create summary metrics and plots
    create_genotype_summary(base_results_dir, all_results)
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main() 