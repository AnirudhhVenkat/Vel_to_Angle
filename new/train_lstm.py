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
    def __init__(self, X, y, sequence_length=50):
        """Initialize sequence dataset.
        
        Args:
            X: Input features array
            y: Target values array
            sequence_length: Length of sequence needed for prediction
        """
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.sequence_length = sequence_length
        
        # Calculate trial size after filtering
        context_frames = sequence_length - 1
        start_frame = 400 - context_frames
        end_frame = 1000
        self.trial_size = end_frame - start_frame + 1  # Include end frame
        self.num_trials = len(self.X) // self.trial_size
        
        # Verify data length
        if len(self.X) != self.num_trials * self.trial_size:
            raise ValueError(
                f"Data length {len(self.X)} is not divisible by "
                f"filtered trial size {self.trial_size}"
            )
        
        # Calculate prediction range
        self.context_frames = context_frames
        self.start_predict = context_frames  # First frame we can predict (frame 400 in original data)
        self.end_predict = self.trial_size - 1  # Last frame (frame 1000 in original data)
        self.frames_to_predict = self.end_predict - self.start_predict + 1
        
        print(f"\nSequenceDataset initialized:")
        print(f"Number of trials: {self.num_trials}")
        print(f"Frames per trial: {self.trial_size}")
        print(f"Context frames: {self.context_frames}")
        print(f"Sequence length: {sequence_length}")
        print(f"Start predict frame (relative to filtered): {self.start_predict}")
        print(f"End predict frame (relative to filtered): {self.end_predict}")
        print(f"Frames to predict per trial: {self.frames_to_predict}")
        print(f"Total sequences possible: {self.__len__()}")
    
    def __len__(self):
        """Return total number of sequences we can create."""
        return self.frames_to_predict * self.num_trials
    
    def __getitem__(self, idx):
        """Get a sequence and its target."""
        # Convert flat index to trial and frame indices
        trial_idx = idx // self.frames_to_predict
        frame_offset = idx % self.frames_to_predict
        
        # Calculate target frame (relative to trial start in filtered data)
        target_frame = self.start_predict + frame_offset
        
        # Calculate indices in the full dataset
        trial_start = trial_idx * self.trial_size
        sequence_end = trial_start + target_frame + 1  # +1 because target_frame is inclusive
        sequence_start = sequence_end - self.sequence_length
        
        # Extract sequence and target
        sequence = self.X[sequence_start:sequence_end]
        target = self.y[sequence_end - 1]
        
        return torch.FloatTensor(sequence), torch.FloatTensor(target)

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

class ZScoreScaler:
    """Z-score normalization scaler"""
    def __init__(self, means=None, stds=None, feature_names=None):
        self.means = means
        self.stds = stds
        self.feature_names = feature_names
    
    def fit(self, X):
        """Fit the scaler to the data."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        # Handle constant features
        self.stds[self.stds == 0] = 1
        return self
    
    def transform(self, X):
        """Transform the data."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return (X - self.means) / self.stds
    
    def fit_transform(self, X):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """Convert back to original scale."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return X * self.stds + self.means

def filter_trial_frames(X, y, sequence_length):
    """Filter frames to include context before frame 400."""
    context_frames = sequence_length - 1
    start_frame = 400 - context_frames  # Start earlier to have context
    end_frame = 1000
    original_trial_size = 1400
    filtered_trial_size = end_frame - start_frame + 1  # Include end frame
    
    filtered_X = []
    filtered_y = []
    
    # Process each trial
    num_trials = len(X) // original_trial_size
    print(f"\nProcessing {num_trials} trials:")
    print(f"Original trial size: {original_trial_size}")
    print(f"Start frame (with context): {start_frame}")
    print(f"End frame: {end_frame}")
    print(f"Context frames needed: {context_frames}")
    print(f"Filtered trial size: {filtered_trial_size}")
    
    for trial in range(num_trials):
        trial_start = trial * original_trial_size
        trial_end = (trial + 1) * original_trial_size
        
        # Extract trial data
        trial_X = X[trial_start:trial_end]
        trial_y = y[trial_start:trial_end]
        
        # Extract frames from start_frame to end_frame (inclusive)
        filtered_trial_X = trial_X[start_frame:end_frame + 1]  # +1 to include end frame
        filtered_trial_y = trial_y[start_frame:end_frame + 1]  # +1 to include end frame
        
        filtered_X.append(filtered_trial_X)
        filtered_y.append(filtered_trial_y)
    
    # Combine all trials
    filtered_X = np.concatenate(filtered_X)
    filtered_y = np.concatenate(filtered_y)
    
    print(f"\nFiltered data shapes:")
    print(f"X: {filtered_X.shape}")
    print(f"y: {filtered_y.shape}")
    
    return filtered_X, filtered_y

def prepare_data(data_path, input_features, output_features, sequence_length, genotype='ALL'):
    """Load and prepare data for training."""
    print("\nPreparing data...")
    
    # Load data with more robust parsing
    try:
        df = pd.read_csv(data_path, engine='python')
        print(f"Loaded data with shape: {df.shape}")
    except Exception as e:
        print(f"Error loading CSV with default settings: {e}")
        print("Trying alternative loading method...")
        try:
            df = pd.read_csv(data_path, engine='python', encoding='utf-8', on_bad_lines='skip')
            print(f"Successfully loaded data with shape: {df.shape}")
        except Exception as e:
            print(f"Failed to load data: {e}")
            raise
    
    # Filter for specific genotype if requested
    if genotype != 'ALL':
        df = df[df['genotype'] == genotype].copy()
        print(f"Filtered for {genotype} genotype: {df.shape}")
    
    # Create trial IDs based on frame numbers
    trial_size = 1400
    num_trials = len(df) // trial_size
    print(f"\nInitial number of trials: {num_trials}")
    print(f"Total frames: {len(df)}")
    print(f"Remainder frames: {len(df) % trial_size}")
    
    # Keep only complete trials
    complete_trials_data = df.iloc[:num_trials * trial_size].copy()
    print(f"Keeping only complete trials: {len(complete_trials_data)} frames")
    
    # Create trial IDs
    complete_trials_data['trial_id'] = np.repeat(np.arange(num_trials), trial_size)
    
    # Calculate split sizes
    train_size = int(0.7 * num_trials)
    val_size = int(0.15 * num_trials)
    test_size = num_trials - train_size - val_size
    
    print(f"\nSplitting data by trials:")
    print(f"Train: {train_size} trials")
    print(f"Validation: {val_size} trials")
    print(f"Test: {test_size} trials")
    
    # Create random permutation of trial indices
    np.random.seed(42)  # For reproducibility
    trial_indices = np.random.permutation(num_trials)
    
    # Split trial indices into train/val/test
    train_trials = trial_indices[:train_size]
    val_trials = trial_indices[train_size:train_size + val_size]
    test_trials = trial_indices[train_size + val_size:]
    
    print("\nTrial assignments:")
    print(f"Training trials: {sorted(train_trials)}")
    print(f"Validation trials: {sorted(val_trials)}")
    print(f"Test trials: {sorted(test_trials)}")
    
    # Create masks for each split
    train_mask = np.zeros(len(complete_trials_data), dtype=bool)
    val_mask = np.zeros(len(complete_trials_data), dtype=bool)
    test_mask = np.zeros(len(complete_trials_data), dtype=bool)
    
    # Assign trials to splits using the random indices
    for trial in train_trials:
        start_idx = trial * trial_size
        end_idx = (trial + 1) * trial_size
        train_mask[start_idx:end_idx] = True
    
    for trial in val_trials:
        start_idx = trial * trial_size
        end_idx = (trial + 1) * trial_size
        val_mask[start_idx:end_idx] = True
    
    for trial in test_trials:
        start_idx = trial * trial_size
        end_idx = (trial + 1) * trial_size
        test_mask[start_idx:end_idx] = True
    
    # Split data
    train_data = complete_trials_data[train_mask].copy()
    val_data = complete_trials_data[val_mask].copy()
    test_data = complete_trials_data[test_mask].copy()
    
    # Calculate moving averages for velocities within each split separately
    base_velocities = ['x_vel', 'y_vel', 'z_vel']
    
    def calculate_moving_averages(data):
        """Calculate moving averages for a dataset"""
        data = data.copy()
        for window in [5, 10, 20]:
            for vel in base_velocities:
                # Calculate moving average for each trial separately
                ma_values = []
                for trial in data['trial_id'].unique():
                    trial_data = data[data['trial_id'] == trial][vel]
                    # Calculate moving average and handle edges
                    ma = trial_data.rolling(window=window, center=True, min_periods=1).mean()
                    ma_values.extend(ma.tolist())
                data[f'{vel}_ma{window}'] = ma_values
        return data
    
    # Calculate moving averages for each split
    train_data = calculate_moving_averages(train_data)
    val_data = calculate_moving_averages(val_data)
    test_data = calculate_moving_averages(test_data)
    
    # Define input features (only velocities)
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
    
    print(f"\nFeature Information:")
    print(f"Input features ({len(input_features)}):")
    for feat in input_features:
        print(f"  - {feat}")
        if feat in train_data.columns:
            print(f"    NaN count in training: {train_data[feat].isna().sum()}")
    
    print(f"\nOutput features ({len(output_features)}):")
    for feat in output_features:
        print(f"  - {feat}")
    
    # Extract features and targets
    X_train = train_data[input_features].values
    y_train = train_data[output_features].values
    X_val = val_data[input_features].values
    y_val = val_data[output_features].values
    X_test = test_data[input_features].values
    y_test = test_data[output_features].values
    
    # Scale the data
    X_scaler = ZScoreScaler()
    y_scaler = ZScoreScaler()
    
    # Fit scalers on training data only
    X_train = X_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train)
    
    # Transform validation and test data
    X_val = X_scaler.transform(X_val)
    y_val = y_scaler.transform(y_val)
    X_test = X_scaler.transform(X_test)
    y_test = y_scaler.transform(y_test)
    
    print("\nData split sizes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"X_test: {X_test.shape}")
    
    # Filter frames to include context before frame 400
    context_frames = sequence_length - 1
    start_frame = 400 - context_frames  # Start earlier to have context
    end_frame = 1000
    filtered_trial_size = end_frame - start_frame + 1  # Size after filtering
    
    print(f"\nFiltering frames {start_frame}-{end_frame} (includes {context_frames} context frames)")
    
    # Filter each split
    X_train_filtered, y_train_filtered = filter_trial_frames(X_train, y_train, sequence_length)
    X_val_filtered, y_val_filtered = filter_trial_frames(X_val, y_val, sequence_length)
    X_test_filtered, y_test_filtered = filter_trial_frames(X_test, y_test, sequence_length)
    
    print("\nFiltered data sizes:")
    print(f"X_train: {X_train_filtered.shape}")
    print(f"X_val: {X_val_filtered.shape}")
    print(f"X_test: {X_test_filtered.shape}")
    
    # Create sequence datasets
    train_dataset = SequenceDataset(X_train_filtered, y_train_filtered, sequence_length)
    val_dataset = SequenceDataset(X_val_filtered, y_val_filtered, sequence_length)
    test_dataset = SequenceDataset(X_test_filtered, y_test_filtered, sequence_length)
    
    return train_dataset, val_dataset, test_dataset, X_scaler, y_scaler

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=20):
    """Train the LSTM model."""
    model = model.to(device)
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    
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
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train MAE: {avg_train_loss:.4f}")
        print(f"  Val MAE: {avg_val_loss:.4f}")
        
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
    """Evaluate the trained model."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    
    # Print shapes for debugging
    print(f"\nRaw predictions shape: {predictions.shape}")
    print(f"Raw targets shape: {targets.shape}")
    
    # Calculate the actual number of trials based on the data size
    # Each trial should have 600 frames (frames 400-1000)
    frames_per_trial = 600
    num_trials = len(predictions) // frames_per_trial
    
    # Ensure we have complete trials
    total_frames = num_trials * frames_per_trial
    predictions = predictions[:total_frames]
    targets = targets[:total_frames]
    
    print(f"\nReshaping arrays:")
    print(f"Number of trials: {num_trials}")
    print(f"Frames per trial: {frames_per_trial}")
    print(f"Total frames to use: {total_frames}")
    
    # Reshape predictions and targets
    try:
        predictions = predictions.reshape(num_trials, frames_per_trial, -1)
        targets = targets.reshape(num_trials, frames_per_trial, -1)
        print(f"Successfully reshaped arrays to: {predictions.shape}")
    except ValueError as e:
        print(f"Error reshaping arrays: {e}")
        print(f"Predictions size: {predictions.size}")
        print(f"Target shape: {targets.shape}")
        raise
    
    # Inverse transform predictions and targets
    predictions_original = output_scaler.inverse_transform(predictions.reshape(-1, predictions.shape[-1]))
    targets_original = output_scaler.inverse_transform(targets.reshape(-1, targets.shape[-1]))
    
    # Reshape back to (num_trials, frames_per_trial, num_features)
    predictions_original = predictions_original.reshape(predictions.shape)
    targets_original = targets_original.reshape(targets.shape)
    
    # Save predictions and targets
    predictions_dir = save_dir / 'predictions'
    predictions_dir.mkdir(exist_ok=True)
    
    # Save all trial predictions in one file
    np.savez(predictions_dir / 'trial_predictions.npz',
             predictions=predictions_original,
             targets=targets_original,
             output_features=output_features,
             frames=np.arange(400, 1000))  # Original frame numbers
    
    # Calculate metrics for each feature
    metrics = {}
    for i, feature in enumerate(output_features):
        # Calculate MAE and RMSE
        mae = np.mean(np.abs(predictions_original[:, :, i] - targets_original[:, :, i]))
        rmse = np.sqrt(np.mean((predictions_original[:, :, i] - targets_original[:, :, i])**2))
        
        metrics[feature] = {
            'mae': float(mae),
            'rmse': float(rmse)
        }
    
    # Save metrics to JSON
    metrics_file = save_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Create plots
    plots_dir = save_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Create PDF for all predictions
    with PdfPages(plots_dir / 'predictions.pdf') as pdf:
        # First page: Summary metrics
        plt.figure(figsize=(12, 8))
        plt.axis('off')
        plt.text(0.1, 0.95, 'Model Prediction Summary', fontsize=16, fontweight='bold')
        
        y_pos = 0.85
        for feature, feature_metrics in metrics.items():
            plt.text(0.1, y_pos, f"\n{feature}:", fontsize=12, fontweight='bold')
            y_pos -= 0.05
            plt.text(0.1, y_pos, f"  MAE: {feature_metrics['mae']:.4f}°")
            y_pos -= 0.03
            plt.text(0.1, y_pos, f"  RMSE: {feature_metrics['rmse']:.4f}°")
            y_pos -= 0.05
        
        pdf.savefig()
        plt.close()
        
        # Plot each trial
        for trial in range(num_trials):
            # Create a figure with subplots for each feature
            fig = plt.figure(figsize=(15, 10))
            plt.suptitle(f'Trial {trial + 1} Predictions', fontsize=16, y=0.95)
            
            # Calculate number of rows and columns for subplots
            n_features = len(output_features)
            n_cols = 2
            n_rows = (n_features + 1) // 2
            
            # Plot each feature
            for i, feature in enumerate(output_features):
                ax = plt.subplot(n_rows, n_cols, i + 1)
                
                # Get predictions and targets for this trial and feature
                trial_pred = predictions_original[trial, :, i]
                trial_target = targets_original[trial, :, i]
                
                # Create time axis (frames 400-1000)
                x_axis = np.arange(400, 1000)
                
                # Plot predictions and targets
                ax.plot(x_axis, trial_target, 'b-', label='Actual', alpha=0.7)
                ax.plot(x_axis, trial_pred, 'r-', label='Predicted', alpha=0.7)
                
                # Calculate trial-specific metrics
                mae = np.mean(np.abs(trial_pred - trial_target))
                rmse = np.sqrt(np.mean((trial_pred - trial_target)**2))
                
                ax.set_title(f'{feature}\nMAE: {mae:.2f}°, RMSE: {rmse:.2f}°')
                ax.set_xlabel('Frame')
                ax.set_ylabel('Angle (degrees)')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close()
            
            # Print progress
            if (trial + 1) % 10 == 0:
                print(f"Processed {trial + 1}/{num_trials} trials")
    
    print(f"\nPrediction plots saved to: {plots_dir / 'predictions.pdf'}")
    return metrics

def objective(trial, data_loaders, input_features, output_features, device, leg_prefix):
    """Optuna objective function for hyperparameter optimization."""
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
    
    # Use MAE loss
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    train_loader, val_loader = data_loaders
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(50):  # Maximum epochs for hyperparameter search
        # Training phase
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        
        # Early stopping based on validation MAE
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:  # Early stopping patience
                break
        
        # Report intermediate value
        trial.report(val_loss, epoch)
        
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return best_val_loss

def main():
    # Configuration
    base_config = {
        'data_path': "Z:/Divya/TEMP_transfers/toAni/BPN_P9LT_P9RT_flyCoords.csv",
        'sequence_length': 50,
        'hidden_size': 256,
        'num_layers': 4,
        'dropout': 0.1,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 200,
        'patience': 20
    }
    
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
        'z_vel'
    ]
    
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
    
    # Create base directories
    results_dir = Path('lstm_results')
    results_dir.mkdir(exist_ok=True)
    print("\nCreated base results directory:", results_dir)
    
    # Create error log file
    error_log = results_dir / 'error_log.txt'
    with open(error_log, 'w') as f:
        f.write(f"Training started at {datetime.now()}\n")
        f.write("-" * 80 + "\n\n")
    
    # Train model for each leg
    for leg_name, leg_info in legs.items():
        try:
            print(f"\n{'='*80}")
            print(f"Training model for {leg_name} leg")
            print(f"{'='*80}")
            
            # Create leg-specific directories
            leg_dir = results_dir / leg_name
            models_dir = leg_dir / 'models'
            plots_dir = leg_dir / 'plots'
            
            leg_dir.mkdir(exist_ok=True)
            models_dir.mkdir(exist_ok=True)
            plots_dir.mkdir(exist_ok=True)
            
            print(f"\nCreated directory structure for {leg_name}:")
            print(f"- {leg_dir}")
            print(f"- {models_dir}")
            print(f"- {plots_dir}")
            
            # Use leg-specific output features
            output_features = leg_info['angles']
            
            print(f"\nFeatures for {leg_name}:")
            print(f"Input features ({len(input_features)}):")
            for feat in input_features:
                print(f"  - {feat}")
            print(f"\nOutput features ({len(output_features)}):")
            for feat in output_features:
                print(f"  - {feat}")
            
            # Prepare data for this leg
            train_dataset, val_dataset, test_dataset, feature_scaler, target_scaler = prepare_data(
                data_path=base_config['data_path'],
                input_features=input_features,
                output_features=output_features,
                sequence_length=base_config['sequence_length']
            )
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=base_config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=base_config['batch_size'])
            test_loader = DataLoader(test_dataset, batch_size=base_config['batch_size'])
            
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
            criterion = nn.L1Loss()  # Use MAE loss for stable training
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
            
            print(f"\nTraining completed for {leg_name}. Best validation MAE: {best_val_loss:.4f}")
            
            # Load best model
            model.load_state_dict(best_model_state)
            
            # Evaluate model
            metrics = evaluate_model(
                model,
                test_loader,
                criterion,
                device,
                output_features,
                target_scaler,
                plots_dir
            )
            
            # Print results
            print(f"\nFinal Test Results for {leg_name}:")
            for angle, angle_metrics in metrics.items():
                print(f"\n{angle}:")
                for metric, value in angle_metrics.items():
                    print(f"  {metric}: {value:.4f}")
            
            # Save model and results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_save_path = models_dir / f'{leg_name}_lstm_{timestamp}.pth'
            
            torch.save({
                'model_state_dict': best_model_state,
                'config': base_config,
                'input_features': input_features,
                'output_features': output_features,
                'feature_scaler': feature_scaler,
                'target_scaler': target_scaler,
                'metrics': metrics,
                'best_val_loss': best_val_loss
            }, model_save_path)
            
            print(f"\nModel and results saved to: {model_save_path}")
            
            # Save summary metrics
            summary_file = leg_dir / 'metrics_summary.txt'
            with open(summary_file, 'w') as f:
                f.write(f"Summary for {leg_name} leg\n")
                f.write("="*50 + "\n\n")
                f.write(f"Best validation MAE: {best_val_loss:.4f}\n")
            
            print(f"\nSummary metrics saved to: {summary_file}")
            
            # Log successful completion
            with open(error_log, 'a') as f:
                f.write(f"Successfully completed training for {leg_name} at {datetime.now()}\n")
                f.write(f"Best validation MAE: {best_val_loss:.4f}\n")
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