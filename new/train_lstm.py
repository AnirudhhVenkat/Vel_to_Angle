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

class SequenceDataset(Dataset):
    """Dataset for LSTM training with sequences"""
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
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Get sequence start index
        sequence_start = self.valid_indices[idx]
        sequence_end = sequence_start + self.sequence_length
        
        # Get sequence of inputs and corresponding target
        X_sequence = self.X[sequence_start:sequence_end]
        y_target = self.y[sequence_end-1]  # Target is the last frame in sequence
        
        return X_sequence, y_target

class LSTMPredictor(nn.Module):
    """LSTM model for predicting joint angles"""
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True  # Input shape: [batch, seq, features]
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        print("\nModel Architecture:")
        print(f"Input size: {input_size}")
        print(f"Hidden size: {hidden_size}")
        print(f"Number of layers: {num_layers}")
        print(f"Output size: {output_size}")
        print(f"Dropout: {dropout}")
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        # out shape: [batch, seq, hidden_size]
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output from the last time step
        out = self.fc(out[:, -1, :])
        return out

class ZScoreScaler:
    """Custom Z-score scaler that maintains feature names and scaling parameters"""
    def __init__(self, means=None, stds=None, feature_names=None):
        self.means = means
        self.stds = stds
        self.feature_names = feature_names
    
    def fit(self, X):
        """Calculate means and standard deviations"""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            self.means = X.mean().to_dict()
            self.stds = X.std().to_dict()
        else:
            if self.feature_names is None:
                raise ValueError("Feature names must be provided for numpy array input")
            self.means = {feat: np.mean(X[:, i]) for i, feat in enumerate(self.feature_names)}
            self.stds = {feat: np.std(X[:, i]) for i, feat in enumerate(self.feature_names)}
    
    def transform(self, X):
        """Z-score normalize the data"""
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame({col: (X[col] - self.means[col]) / self.stds[col] 
                               for col in X.columns}, columns=X.columns)
        else:
            return np.column_stack([(X[:, i] - self.means[feat]) / self.stds[feat]
                                  for i, feat in enumerate(self.feature_names)])
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        """Convert z-scores back to original scale"""
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame({col: (X[col] * self.stds[col]) + self.means[col]
                               for col in X.columns}, columns=X.columns)
        else:
            return np.column_stack([(X[:, i] * self.stds[feat]) + self.means[feat]
                                  for i, feat in enumerate(self.feature_names)])

def filter_trial_frames(X, y, sequence_length):
    """Filter frames to only include frames 400-1000 from each trial, with context"""
    # Calculate number of frames needed for context
    context_frames = sequence_length - 1
    
    # Calculate frames per trial after filtering (1000 - 400 = 600)
    frames_per_trial = 600
    
    # Calculate total sequence length (context + prediction frames)
    total_sequence = context_frames + frames_per_trial
    
    # Calculate number of trials
    num_trials = len(X) // (frames_per_trial + context_frames)
    
    # Initialize arrays for filtered data
    X_filtered = []
    y_filtered = []
    
    for trial in range(num_trials):
        # Calculate start index for this trial's context
        start_context = trial * (frames_per_trial + context_frames)
        
        # Get trial data including context
        X_trial = X[start_context:start_context + total_sequence]
        y_trial = y[start_context:start_context + total_sequence]
        
        X_filtered.append(X_trial)
        y_filtered.append(y_trial)
    
    # Concatenate all trials
    X_filtered = np.concatenate(X_filtered)
    y_filtered = np.concatenate(y_filtered)
    
    return X_filtered, y_filtered

def prepare_data(data_path, input_features, output_features, sequence_length, genotype='ALL'):
    """Prepare sequences of data for LSTM"""
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
    
    # Define input features (only velocities)
    velocity_features = [
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
        'z_vel'
    ]
    
    # Verify all input features are velocity-based
    for feature in input_features:
        if feature not in velocity_features:
            print(f"Warning: Non-velocity feature {feature} will be ignored")
    input_features = velocity_features
    
    print("\nFeature Information:")
    print(f"Input features ({len(input_features)}):")
    for feat in input_features:
        print(f"  - {feat}")
        if feat in df.columns:
            print(f"    NaN count: {df[feat].isna().sum()}")
    
    print(f"\nOutput features ({len(output_features)}):")
    for feat in output_features:
        print(f"  - {feat}")
    
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
    
    return (train_loader, val_loader, test_loader), (X_scaler, y_scaler), (input_features, output_features)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=20):
    """Train the LSTM model."""
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'mae': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_pbar.set_postfix({'mae': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train MAE: {avg_train_loss:.4f}")
        print(f"  Val MAE: {avg_val_loss:.4f}")
        print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            patience_counter = 0
            print(f"  New best model! Val MAE: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    return best_model, best_val_loss

def evaluate_model(model, test_loader, criterion, device, output_features, output_scaler, save_dir):
    """Evaluate the trained LSTM model"""
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

def main():
    """Main function to train and evaluate the LSTM model."""
    # Configuration
    config = {
        'data_path': "Z:/Divya/TEMP_transfers/toAni/BPN_P9LT_P9RT_flyCoords.csv",
        'batch_size': 64,
        'learning_rate': 1e-3,
        'num_epochs': 200,
        'patience': 20,
        'sequence_length': 50,
        'hidden_size': 256,
        'num_layers': 2,
        'dropout': 0.2
    }
    
    # Process each leg
    legs = {
        'R1': {
            'angles': ['R1A_flex', 'R1A_rot', 'R1A_abduct',
                      'R1B_flex', 'R1B_rot',
                      'R1C_flex', 'R1C_rot',
                      'R1D_flex']
        },
        'L1': {
            'angles': ['L1A_flex', 'L1A_rot', 'L1A_abduct',
                      'L1B_flex', 'L1B_rot',
                      'L1C_flex', 'L1C_rot',
                      'L1D_flex']
        },
        'R2': {
            'angles': ['R2A_flex', 'R2A_rot', 'R2A_abduct',
                      'R2B_flex', 'R2B_rot',
                      'R2C_flex', 'R2C_rot',
                      'R2D_flex']
        },
        'L2': {
            'angles': ['L2A_flex', 'L2A_rot', 'L2A_abduct',
                      'L2B_flex', 'L2B_rot',
                      'L2C_flex', 'L2C_rot',
                      'L2D_flex']
        },
        'R3': {
            'angles': ['R3A_flex', 'R3A_rot', 'R3A_abduct',
                      'R3B_flex', 'R3B_rot',
                      'R3C_flex', 'R3C_rot',
                      'R3D_flex']
        },
        'L3': {
            'angles': ['L3A_flex', 'L3A_rot', 'L3A_abduct',
                      'L3B_flex', 'L3B_rot',
                      'L3C_flex', 'L3C_rot',
                      'L3D_flex']
        }
    }
    
    # Create directories
    results_dir = Path('lstm_results')
    models_dir = results_dir / 'models'
    plots_dir = results_dir / 'plots'
    
    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    
    # Process each leg
    for leg_prefix, leg_config in legs.items():
        print(f"\n{'='*80}")
        print(f"Processing {leg_prefix} leg")
        print(f"{'='*80}")
        
        # Define input features
        base_features = [
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
            'z_vel'
        ]
        
        # Input features: only velocities
        input_features = base_features
        
        # Output features: 8 angles of R1 leg
        output_features = [
            'R1A_flex', 'R1A_rot', 'R1A_abduct',
            'R1B_flex', 'R1B_rot',
            'R1C_flex', 'R1C_rot',
            'R1D_flex'
        ]
        
        # Prepare data
        loaders, scalers, features = prepare_data(
            config['data_path'],
            input_features,
            leg_config['angles'],
            config['sequence_length']
        )
        train_loader, val_loader, test_loader = loaders
        X_scaler, y_scaler = scalers
        
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMPredictor(
            input_size=len(input_features),
            hidden_size=config['hidden_size'],
            output_size=len(leg_config['angles']),
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(device)
        
        # Create optimizer and criterion
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        criterion = nn.L1Loss()  # MAE loss
        
        # Train model
        best_model_state, best_val_loss = train_model(
            model, train_loader, val_loader, criterion,
            optimizer, config['num_epochs'], device,
            config['patience']
        )
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Evaluate model
        leg_plots_dir = plots_dir / leg_prefix
        leg_plots_dir.mkdir(exist_ok=True)
        
        metrics = evaluate_model(
            model, test_loader, criterion, device,
            leg_config['angles'], y_scaler, leg_plots_dir
        )
        
        # Save model and results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = models_dir / f'lstm_{leg_prefix}_{timestamp}.pth'
        
        torch.save({
            'model_state_dict': best_model_state,
            'config': config,
            'input_features': input_features,
            'output_features': leg_config['angles'],
            'X_scaler': X_scaler,
            'y_scaler': y_scaler,
            'metrics': metrics
        }, save_path)
        
        print(f"\nModel and results saved to: {save_path}")
        
        # Print metrics
        print("\nFinal Results:")
        print(f"Test Loss: {metrics['test_loss']:.4f}")
        
        for feature, feature_metrics in metrics['metrics_by_feature'].items():
            print(f"\n{feature}:")
            print(f"  MAE (z-scored): {feature_metrics['mae_z']:.4f}")
            print(f"  RMSE (z-scored): {feature_metrics['rmse_z']:.4f}")
            print(f"  R² (z-scored): {feature_metrics['r2_z']:.4f}")
            print(f"  MAE (raw): {feature_metrics['mae_raw']:.4f}°")
            print(f"  RMSE (raw): {feature_metrics['rmse_raw']:.4f}°")
            print(f"  R² (raw): {feature_metrics['r2_raw']:.4f}")

if __name__ == "__main__":
    main() 