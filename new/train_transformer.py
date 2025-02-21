import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime
import math
import json
import traceback

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=8, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        self.model_type = 'Transformer'
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Ensure hidden size is divisible by number of heads
        assert hidden_size % nhead == 0, f"Hidden size ({hidden_size}) must be divisible by number of heads ({nhead})"
        
        # Initial normalization of input
        self.input_norm = nn.LayerNorm(input_size)
        
        # Input projection with activation and dropout
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Initialize parameters
        self._init_parameters()
        
        print("\nModel Architecture:")
        print(f"Input size: {input_size}")
        print(f"Hidden size: {hidden_size}")
        print(f"Number of heads: {nhead}")
        print(f"Number of layers: {num_layers}")
        print(f"Output size: {output_size}")
        
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
    
    def forward(self, src):
        # Input shape: (batch, seq_len, input_size)
        
        # Normalize input
        x = self.input_norm(src)
        
        # Project input to transformer dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create padding mask for any zero vectors in input
        padding_mask = torch.all(src == 0, dim=-1)
        
        # Apply transformer encoder with padding mask
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Project to output dimension
        # Take the last sequence element for prediction
        x = x[:, -1]  # Shape: (batch, hidden_size)
        output = self.output_projection(x)  # Shape: (batch, output_size)
        
        return output

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
        
        # Each trial is 1400 frames
        self.trial_size = 1400
        self.num_trials = len(self.X) // self.trial_size
        
        # We want to predict frames 400-1000 in each trial
        self.start_predict = 400
        self.end_predict = 1000
        self.frames_to_predict = self.end_predict - self.start_predict + 1  # 601 frames (inclusive)
        
        # Verify data length
        if len(self.X) != self.num_trials * self.trial_size:
            raise ValueError(
                f"Data length {len(self.X)} is not divisible by "
                f"trial_size {self.trial_size}"
            )
        
        print(f"\nSequenceDataset initialized:")
        print(f"Number of trials: {self.num_trials}")
        print(f"Frames per trial: {self.trial_size}")
        print(f"Sequence length: {sequence_length}")
        print(f"Prediction range: frames {self.start_predict}-{self.end_predict}")
        print(f"Frames to predict per trial: {self.frames_to_predict}")
        print(f"Total sequences possible: {self.__len__()}")
    
    def __len__(self):
        """Return total number of sequences we can create.
        For each trial, we can create one sequence for each frame we want to predict (400-1000).
        """
        return self.frames_to_predict * self.num_trials
    
    def __getitem__(self, idx):
        """Get a sequence and its target.
        
        Args:
            idx: Index of the sequence to get
        
        Returns:
            sequence: Input sequence of shape (sequence_length, num_features)
            target: Target value for the last frame in sequence
        """
        # Convert flat index to trial and frame indices
        trial_idx = idx // self.frames_to_predict
        frame_offset = idx % self.frames_to_predict
        
        # Calculate the target frame (400-1000) for this sequence
        target_frame = self.start_predict + frame_offset
        
        # Calculate indices in the full dataset
        trial_start = trial_idx * self.trial_size
        sequence_end = trial_start + target_frame
        sequence_start = sequence_end - self.sequence_length
        
        # Extract sequence and target
        sequence = self.X[sequence_start:sequence_end]
        target = self.y[sequence_end - 1]  # Target is the last frame in sequence
        
        return torch.FloatTensor(sequence), torch.FloatTensor(target)

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
    start_frame = 400 - context_frames
    end_frame = 1000
    frames_per_trial = 1400
    
    filtered_X = []
    filtered_y = []
    
    # Process each trial
    num_trials = len(X) // frames_per_trial
    for trial in range(num_trials):
        trial_start = trial * frames_per_trial
        trial_end = (trial + 1) * frames_per_trial
        
        # Extract trial data
        trial_X = X[trial_start:trial_end]
        trial_y = y[trial_start:trial_end]
        
        # Extract frames from start_frame to end_frame
        # This includes context frames before frame 400
        filtered_trial_X = trial_X[start_frame:end_frame]
        filtered_trial_y = trial_y[start_frame:end_frame]
        
        filtered_X.append(filtered_trial_X)
        filtered_y.append(filtered_trial_y)
    
    # Combine all trials
    filtered_X = np.concatenate(filtered_X)
    filtered_y = np.concatenate(filtered_y)
    
    return filtered_X, filtered_y

def prepare_data(data_path, input_features, output_features, sequence_length, genotype='ALL'):
    """Load and prepare data for training."""
    print("\nPreparing data...")
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded data with shape: {df.shape}")
    
    # Filter for specific genotype if requested
    if genotype != 'ALL':
        df = df[df['genotype'] == genotype].copy()
        print(f"Filtered for {genotype} genotype: {df.shape}")
    
    # Create trial IDs based on frame numbers (guaranteed 1400 frames per trial)
    trial_size = 1400
    num_trials = len(df) // trial_size
    print(f"\nNumber of trials detected: {num_trials}")
    print(f"Frames per trial: {trial_size}")
    
    # Create trial IDs
    df['trial_id'] = np.repeat(np.arange(num_trials), trial_size)
    
    # Calculate moving averages for velocities within each trial
    base_velocities = ['x_vel', 'y_vel', 'z_vel']
    for window in [5, 10, 20]:
        for vel in base_velocities:
            # Calculate moving average for each trial separately
            ma_values = []
            for trial in df['trial_id'].unique():
                trial_data = df[df['trial_id'] == trial][vel]
                # Calculate moving average and handle edges
                ma = trial_data.rolling(window=window, center=True, min_periods=1).mean()
                ma_values.extend(ma.tolist())
            df[f'{vel}_ma{window}'] = ma_values
    
    print("Moving averages calculated.")
    
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
    
    # Input features: base velocities only
    input_features = base_features
    
    # Extract features and targets
    X = df[input_features].values
    y = df[output_features].values
    
    print(f"\nFeature Information:")
    print(f"Input features ({len(input_features)}):")
    for feat in input_features:
        print(f"  - {feat}")
        if feat in df.columns:
            print(f"    NaN count: {df[feat].isna().sum()}")
    
    print(f"\nOutput features ({len(output_features)}):")
    for feat in output_features:
        print(f"  - {feat}")
    
    # Split into train, validation, and test sets by trials
    train_size = int(0.7 * num_trials)
    val_size = int(0.15 * num_trials)
    test_size = num_trials - train_size - val_size
    
    print(f"\nSplitting data by trials:")
    print(f"Train: {train_size} trials")
    print(f"Validation: {val_size} trials")
    print(f"Test: {test_size} trials")
    
    # Create masks for each split
    train_mask = np.zeros(len(df), dtype=bool)
    val_mask = np.zeros(len(df), dtype=bool)
    test_mask = np.zeros(len(df), dtype=bool)
    
    # Assign trials to splits
    for trial in range(num_trials):
        start_idx = trial * trial_size
        end_idx = (trial + 1) * trial_size
        
        if trial < train_size:
            train_mask[start_idx:end_idx] = True
        elif trial < train_size + val_size:
            val_mask[start_idx:end_idx] = True
        else:
            test_mask[start_idx:end_idx] = True
    
    # Scale the data
    X_scaler = ZScoreScaler()
    y_scaler = ZScoreScaler()
    
    # Fit scalers on training data only
    X_train = X_scaler.fit_transform(X[train_mask])
    y_train = y_scaler.fit_transform(y[train_mask])
    
    # Transform validation and test data
    X_val = X_scaler.transform(X[val_mask])
    y_val = y_scaler.transform(y[val_mask])
    X_test = X_scaler.transform(X[test_mask])
    y_test = y_scaler.transform(y[test_mask])
    
    print("\nData split sizes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"X_test: {X_test.shape}")
    
    # Filter frames to include context before frame 400
    context_frames = sequence_length - 1
    start_frame = 400 - context_frames  # Start earlier to have context
    end_frame = 1000
    
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
    """Train the transformer model."""
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
    
    # Calculate metrics for each feature
    metrics = {}
    for i, feature in enumerate(output_features):
        # Calculate MAE and RMSE
        mae = float(np.mean(np.abs(predictions_original[:, :, i] - targets_original[:, :, i])))
        rmse = float(np.sqrt(np.mean((predictions_original[:, :, i] - targets_original[:, :, i])**2)))
        r2 = float(1 - np.sum((targets_original[:, :, i] - predictions_original[:, :, i])**2) / np.sum((targets_original[:, :, i] - targets_original[:, :, i].mean())**2))
        
        metrics[feature] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
    # Save metrics to JSON
    metrics_file = Path(save_dir) / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Create plots directory
    plots_dir = Path(save_dir) / 'plots'
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
            y_pos -= 0.03
            plt.text(0.1, y_pos, f"  R²: {feature_metrics['r2']:.4f}")
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
                r2 = 1 - np.sum((trial_target - trial_pred)**2) / np.sum((trial_target - trial_target.mean())**2)
                
                ax.set_title(f'{feature}\nMAE: {mae:.2f}°, RMSE: {rmse:.2f}°, R²: {r2:.2f}')
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

def main():
    """Main function to train and evaluate transformer models for all 6 legs."""
    # Configuration
    base_config = {
        'data_path': "Z:/Divya/TEMP_transfers/toAni/BPN_P9LT_P9RT_flyCoords.csv",
        'sequence_length': 50,
        'hidden_size': 256,
        'num_heads': 8,
        'num_layers': 4,
        'dropout': 0.1,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 200,
        'patience': 20
    }
    
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
    results_dir = Path('transformer_results')
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
            train_dataset, val_dataset, test_dataset, X_scaler, y_scaler = prepare_data(
                base_config['data_path'],
                input_features,
                output_features,
                base_config['sequence_length']
            )
            
            # Create model for this leg
            model = TransformerPredictor(
                input_size=len(input_features),
                hidden_size=base_config['hidden_size'],
                output_size=len(output_features),
                nhead=base_config['num_heads'],
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
                train_dataset,
                val_dataset,
                criterion,
                optimizer,
                base_config['num_epochs'],
                device,
                base_config['patience']
            )
            
            print(f"\nTraining completed for {leg_name}. Best validation loss: {best_val_loss:.4f}")
            
            # Load best model
            model.load_state_dict(best_model_state)
            
            # Evaluate model
            metrics = evaluate_model(
                model,
                test_dataset,
                criterion,
                device,
                output_features,
                y_scaler,
                leg_dir
            )
            
            # Save model and results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_save_path = models_dir / f'{leg_name}_transformer_{timestamp}.pth'
            
            torch.save({
                'model_state_dict': best_model_state,
                'config': base_config,
                'input_features': input_features,
                'output_features': output_features,
                'X_scaler': X_scaler,
                'y_scaler': y_scaler,
                'metrics': metrics,
                'best_val_loss': best_val_loss
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