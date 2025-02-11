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
        # Convert inputs to float32 numpy arrays
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.sequence_length = sequence_length
        
        # Calculate total frames per trial (from 400-sequence_length to 1000)
        self.frames_per_trial = 1000 - (400 - sequence_length) + 1
        self.num_trials = len(self.X) // self.frames_per_trial
        
        # Calculate number of sequences we can create per trial
        # We can predict one frame for each position from 400 to 1000
        self.sequences_per_trial = 1000 - 400 + 1  # 601 frames (inclusive)
        
        # Verify data length
        if len(self.X) != self.num_trials * self.frames_per_trial:
            raise ValueError(
                f"Data length {len(self.X)} is not divisible by "
                f"frames_per_trial {self.frames_per_trial}"
            )
        
        print(f"\nSequenceDataset initialized:")
        print(f"Number of trials: {self.num_trials}")
        print(f"Frames per trial: {self.frames_per_trial}")
        print(f"Sequence length: {sequence_length}")
        print(f"Context frames before frame 400: {sequence_length - 1}")
        print(f"Prediction range per trial: frames 400-1000")
        print(f"Sequences per trial: {self.sequences_per_trial}")
        print(f"Total sequences possible: {self.__len__()}")
    
    def __len__(self):
        """Return total number of sequences we can create.
        For each trial, we can create one sequence for each frame we want to predict (400-1000).
        """
        return self.sequences_per_trial * self.num_trials
    
    def __getitem__(self, idx):
        """Get a sequence and its target.
        
        Args:
            idx: Index of the sequence to get
        
        Returns:
            sequence: Input sequence of shape (sequence_length, num_features)
            target: Target value for the last frame in sequence
        """
        # Convert flat index to trial and frame indices
        trial_idx = idx // self.sequences_per_trial
        frame_idx = idx % self.sequences_per_trial
        
        # Calculate the actual start index in the trial's data
        trial_start = trial_idx * self.frames_per_trial
        
        # The target frame is 400 + frame_idx (0 to 600)
        target_frame = 400 + frame_idx
        
        # Calculate sequence start and end
        # We need sequence_length frames ending at target_frame
        sequence_end = trial_start + (target_frame - (400 - self.sequence_length))
        sequence_start = sequence_end - self.sequence_length
        
        # Extract sequence and target
        sequence = self.X[sequence_start:sequence_end]
        target = self.y[sequence_end - 1]  # Target is the last frame in sequence
        
        return torch.FloatTensor(sequence), torch.FloatTensor(target)

def filter_trial_frames(X, y, sequence_length):
    """Filter frames to include frames (400-sequence_length) to 1000 from each trial.
    This ensures we have enough context frames to predict starting from frame 400.
    
    Args:
        X: Input features array
        y: Target values array
        sequence_length: Length of sequence needed for prediction
    
    Returns:
        X_filtered: Filtered input features
        y_filtered: Filtered target values
    """
    # Calculate the number of context frames needed
    context_frames = sequence_length - 1
    
    # Calculate start and end frames
    start_frame = 400 - context_frames
    end_frame = 1000
    
    # Calculate indices for each trial (assuming 1400 frames per trial)
    trial_size = 1400
    num_trials = len(X) // trial_size
    
    # Initialize lists to store filtered data
    X_filtered_list = []
    y_filtered_list = []
    
    # Process each trial
    for trial in range(num_trials):
        # Calculate indices for this trial
        trial_start = trial * trial_size + start_frame
        trial_end = trial * trial_size + end_frame + 1  # +1 to include frame 1000
        
        # Extract and store the frames for this trial
        X_filtered_list.append(X[trial_start:trial_end])
        y_filtered_list.append(y[trial_start:trial_end])
    
    # Concatenate all trials
    X_filtered = np.concatenate(X_filtered_list)
    y_filtered = np.concatenate(y_filtered_list)
    
    print(f"\nFiltering summary:")
    print(f"Original shape: {X.shape}")
    print(f"Filtered shape: {X_filtered.shape}")
    print(f"Context frames included: {context_frames}")
    print(f"Frame range per trial: {start_frame}-{end_frame}")
    print(f"Total frames per trial: {end_frame - start_frame + 1}")
    print(f"Number of trials: {num_trials}")
    
    return X_filtered, y_filtered

def prepare_data(data_path, input_features, output_features, sequence_length, genotype='ALL'):
    """Prepare data for training, including proper context frames for prediction.
    
    Args:
        data_path: Path to the data file
        input_features: List of input feature names
        output_features: List of output feature names
        sequence_length: Length of sequence needed for prediction
        genotype: Genotype to filter for (default: 'ALL')
    
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        X_scaler: Fitted scaler for input features
        y_scaler: Fitted scaler for output features
    """
    print("\nPreparing data...")
    print(f"Sequence length: {sequence_length}")
    print(f"Context frames needed: {sequence_length - 1}")
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded data with shape: {df.shape}")
    
    # Filter for genotype if specified
    if genotype != 'ALL':
        df = df[df['genotype'] == genotype]
        print(f"Filtered for {genotype}: {df.shape}")
    
    # Calculate moving averages for velocities
    base_velocities = ['x_vel', 'y_vel', 'z_vel']
    windows = [5, 10, 20]
    
    # Process each trial separately to avoid boundary effects
    trial_size = 1400  # Original trial size
    num_trials = len(df) // trial_size
    
    print("\nCalculating moving averages...")
    for window in windows:
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
    
    print("Moving averages calculated.")
    
    # Extract features and targets
    X = df[input_features].values
    y = df[output_features].values
    
    # Filter frames and include context
    X_filtered, y_filtered = filter_trial_frames(X, y, sequence_length)
    
    # Split into train, validation, and test sets
    # Use trial-based splitting to maintain sequence integrity
    trial_size = 1000 - (400 - sequence_length) + 1  # Includes context frames
    num_trials = len(X_filtered) // trial_size
    
    # Create trial indices and shuffle
    trial_indices = np.arange(num_trials)
    np.random.seed(42)
    np.random.shuffle(trial_indices)
    
    # Split trials (60% train, 20% val, 20% test)
    train_size = int(0.6 * num_trials)
    val_size = int(0.2 * num_trials)
    
    train_trials = trial_indices[:train_size]
    val_trials = trial_indices[train_size:train_size + val_size]
    test_trials = trial_indices[train_size + val_size:]
    
    # Split data based on trial indices
    X_train = np.concatenate([X_filtered[i * trial_size:(i + 1) * trial_size] for i in train_trials])
    y_train = np.concatenate([y_filtered[i * trial_size:(i + 1) * trial_size] for i in train_trials])
    
    X_val = np.concatenate([X_filtered[i * trial_size:(i + 1) * trial_size] for i in val_trials])
    y_val = np.concatenate([y_filtered[i * trial_size:(i + 1) * trial_size] for i in val_trials])
    
    X_test = np.concatenate([X_filtered[i * trial_size:(i + 1) * trial_size] for i in test_trials])
    y_test = np.concatenate([y_filtered[i * trial_size:(i + 1) * trial_size] for i in test_trials])
    
    # Scale the data
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    # Fit scalers on training data only
    X_train = X_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train)
    
    # Transform validation and test data
    X_val = X_scaler.transform(X_val)
    y_val = y_scaler.transform(y_val)
    X_test = X_scaler.transform(X_test)
    y_test = y_scaler.transform(y_test)
    
    print("\nData split summary:")
    print(f"Training: {len(train_trials)} trials ({len(X_train)} frames)")
    print(f"Validation: {len(val_trials)} trials ({len(X_val)} frames)")
    print(f"Test: {len(test_trials)} trials ({len(X_test)} frames)")
    print(f"Frames per trial: {trial_size} (includes {sequence_length - 1} context frames)")
    print(f"Prediction range: frames 400-1000 for each trial")
    
    # Create datasets
    train_dataset = SequenceDataset(X_train, y_train, sequence_length)
    val_dataset = SequenceDataset(X_val, y_val, sequence_length)
    test_dataset = SequenceDataset(X_test, y_test, sequence_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    return train_loader, val_loader, test_loader, X_scaler, y_scaler

def calculate_r2_score(y_true, y_pred):
    """Calculate R² score for multi-dimensional arrays.
    
    Args:
        y_true: Ground truth values of shape (batch_size, num_features)
        y_pred: Predicted values of shape (batch_size, num_features)
    
    Returns:
        Array of R² scores for each feature
    """
    # Convert to numpy if tensors
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Calculate R² for each dimension
    r2_scores = []
    for i in range(y_true.shape[1]):
        numerator = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
        denominator = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
        r2 = 1 - numerator / denominator
        r2_scores.append(r2)
    
    return np.array(r2_scores)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=20):
    """Train the transformer model."""
    model = model.to(device)
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    best_r2 = -float('inf')  # Track best R² value
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        all_train_preds = []
        all_train_targets = []
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Store predictions and targets for R² calculation
            all_train_preds.append(outputs.detach().cpu().numpy())
            all_train_targets.append(targets.detach().cpu().numpy())
            
            # Use MAE for stable training
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'mae_loss': f'{loss.item():.4f}'})
        
        # Calculate R² for training set
        train_preds = np.concatenate(all_train_preds)
        train_targets = np.concatenate(all_train_targets)
        train_r2_scores = calculate_r2_score(train_targets, train_preds)
        train_r2 = np.mean(train_r2_scores)  # Average R² across all dimensions
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_targets = []
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # Store predictions and targets for R² calculation
                all_val_preds.append(outputs.cpu().numpy())
                all_val_targets.append(targets.cpu().numpy())
                
                # Use MAE for stable validation
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_pbar.set_postfix({'mae_loss': f'{loss.item():.4f}'})
        
        # Calculate R² for validation set
        val_preds = np.concatenate(all_val_preds)
        val_targets = np.concatenate(all_val_targets)
        val_r2_scores = calculate_r2_score(val_targets, val_preds)
        val_r2 = np.mean(val_r2_scores)  # Average R² across all dimensions
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train - MAE: {train_loss/len(train_loader):.4f}, R²: {train_r2:.4f}")
        print(f"  Val - MAE: {val_loss/len(val_loader):.4f}, R²: {val_r2:.4f}")
        
        # Use 1 - R² as the loss metric for optimization (since we want to maximize R²)
        curr_val_loss = 1 - val_r2
        
        # Check for improvement
        if curr_val_loss < best_val_loss:
            best_val_loss = curr_val_loss
            best_r2 = val_r2  # Store the actual R² value
            best_model = model.state_dict()
            patience_counter = 0
            print(f"  New best model! Val R²: {val_r2:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation R²: {best_r2:.4f}")
                break
    
    return best_model, best_r2  # Return the actual R² value instead of loss

def evaluate_model(model, test_loader, criterion, device, output_features, output_scaler, save_dir):
    """Evaluate the trained model."""
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    test_loss = 0.0
    
    # Use tqdm for test evaluation
    test_pbar = tqdm(test_loader, desc='Evaluating')
    
    with torch.no_grad():
        for sequences, targets in test_pbar:
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            test_pbar.set_postfix({'mae_loss': f'{loss.item():.4f}'})
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate all predictions and targets
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    
    # Calculate overall R² score
    test_r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - targets.mean()) ** 2)
    print(f"\nTest MAE: {test_loss/len(test_loader):.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Inverse transform predictions and targets
    predictions = output_scaler.inverse_transform(predictions)
    targets = output_scaler.inverse_transform(targets)
    
    # Create plots directory if it doesn't exist
    plots_dir = Path(save_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate metrics for each joint angle
    metrics = {}
    print("\nCalculating metrics for each joint angle...")
    
    # Calculate frames per trial and number of trials
    frames_per_trial = 600  # Full range from 400-1000
    num_trials = len(predictions) // frames_per_trial
    print(f"\nNumber of trials: {num_trials}")
    print(f"Frames per trial: {frames_per_trial}")
    
    for i, feature in enumerate(tqdm(output_features, desc='Joint Angles')):
        pred = predictions[:, i]
        target = targets[:, i]
        
        mae = np.mean(np.abs(pred - target))
        rmse = np.sqrt(np.mean((pred - target) ** 2))
        r2 = 1 - np.sum((target - pred) ** 2) / np.sum((target - target.mean()) ** 2)
        
        metrics[feature] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(target, pred, alpha=0.5)
        plt.plot([target.min(), target.max()],
                [target.min(), target.max()],
                'r--', lw=2)
        
        plt.xlabel(f'Actual {feature}')
        plt.ylabel(f'Predicted {feature}')
        plt.title(f'{feature} Predictions\nMAE: {mae:.3f}, R²: {r2:.3f}')
        
        # Add metrics text box
        plt.text(0.05, 0.95,
                f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r2:.3f}',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / f'{feature}_scatter.png')
        plt.close()
        
        # Create PDF with time series plots for all trials
        with PdfPages(plots_dir / f'{feature}_predictions.pdf') as pdf:
            # Calculate number of rows and columns for subplots
            n_cols = 2
            n_rows = (num_trials + n_cols - 1) // n_cols
            
            # Create figure with subplots
            fig = plt.figure(figsize=(15, 5 * n_rows))
            
            for trial in range(num_trials):
                plt.subplot(n_rows, n_cols, trial + 1)
                
                # Get trial data
                start_idx = trial * frames_per_trial
                end_idx = start_idx + frames_per_trial
                
                trial_pred = pred[start_idx:end_idx]
                trial_target = target[start_idx:end_idx]
                
                # Calculate trial-specific metrics
                trial_mae = np.mean(np.abs(trial_pred - trial_target))
                trial_r2 = 1 - np.sum((trial_target - trial_pred) ** 2) / np.sum((trial_target - trial_target.mean()) ** 2)
                
                # Create frame numbers for x-axis (400 to 1000)
                frame_numbers = np.linspace(400, 1000, frames_per_trial)
                
                # Plot predictions vs actual
                plt.plot(frame_numbers, trial_target, 'b-', label='Actual', alpha=0.7)
                plt.plot(frame_numbers, trial_pred, 'r-', label='Predicted', alpha=0.7)
                
                plt.title(f'Trial {trial+1}\nMAE: {trial_mae:.3f}, R²: {trial_r2:.3f}')
                plt.xlabel('Frame')
                plt.ylabel('Angle (degrees)')
                plt.grid(True, alpha=0.3)
                
                if trial == 0:  # Only show legend for first subplot
                    plt.legend()
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
    
    return metrics, test_r2

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
            'input_coord': 'R-F-CTr_y',
            'angles': ['R1A_flex', 'R1A_rot', 'R1A_abduct', 'R1B_flex', 'R1B_rot', 'R1C_flex', 'R1C_rot', 'R1D_flex']
        },
        'L1': {
            'input_coord': 'L-F-CTr_y',
            'angles': ['L1A_flex', 'L1A_rot', 'L1A_abduct', 'L1B_flex', 'L1B_rot', 'L1C_flex', 'L1C_rot', 'L1D_flex']
        },
        'R2': {
            'input_coord': 'R-M-CTr_y',
            'angles': ['R2A_flex', 'R2A_rot', 'R2A_abduct', 'R2B_flex', 'R2B_rot', 'R2C_flex', 'R2C_rot', 'R2D_flex']
        },
        'L2': {
            'input_coord': 'L-M-CTr_y',
            'angles': ['L2A_flex', 'L2A_rot', 'L2A_abduct', 'L2B_flex', 'L2B_rot', 'L2C_flex', 'L2C_rot', 'L2D_flex']
        },
        'R3': {
            'input_coord': 'R-H-CTr_y',
            'angles': ['R3A_flex', 'R3A_rot', 'R3A_abduct', 'R3B_flex', 'R3B_rot', 'R3C_flex', 'R3C_rot', 'R3D_flex']
        },
        'L3': {
            'input_coord': 'L-H-CTr_y',
            'angles': ['L3A_flex', 'L3A_rot', 'L3A_abduct', 'L3B_flex', 'L3B_rot', 'L3C_flex', 'L3C_rot', 'L3D_flex']
        }
    }
    
    # Create base directories
    results_dir = Path('transformer_results')
    results_dir.mkdir(exist_ok=True)
    print("\nCreated base results directory:", results_dir)
    
    # Train model for each leg
    for leg_name, leg_info in legs.items():
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
        
        # Define input features
        input_features = [
            'x_vel', 'y_vel', 'z_vel',  # Raw velocities
            'x_vel_ma5', 'y_vel_ma5', 'z_vel_ma5',  # 5-frame moving averages
            'x_vel_ma10', 'y_vel_ma10', 'z_vel_ma10',  # 10-frame moving averages
            'x_vel_ma20', 'y_vel_ma20', 'z_vel_ma20',  # 20-frame moving averages
            leg_info['input_coord']  # Leg-specific coordinate
        ]
        
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
        train_loader, val_loader, test_loader, X_scaler, y_scaler = prepare_data(
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
        best_model_state, best_val_r2 = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            base_config['num_epochs'],
            device,
            base_config['patience']
        )
        
        print(f"\nTraining completed for {leg_name}. Best validation R²: {best_val_r2:.4f}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Evaluate model
        metrics, test_r2 = evaluate_model(
            model,
            test_loader,
            criterion,
            device,
            output_features,
            y_scaler,
            plots_dir
        )
        
        # Print results
        print(f"\nFinal Test Results for {leg_name}:")
        print(f"Average Test R²: {np.mean(test_r2):.4f}")
        for angle, angle_metrics in metrics.items():
            print(f"\n{angle}:")
            for metric, value in angle_metrics.items():
                print(f"  {metric}: {value:.4f}")
        
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
            'best_val_r2': best_val_r2,
            'test_r2': test_r2
        }, model_save_path)
        
        print(f"\nModel and results saved to: {model_save_path}")
        
        # Save summary metrics
        summary_file = leg_dir / 'metrics_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Summary for {leg_name} leg\n")
            f.write("="*50 + "\n\n")
            f.write(f"Best validation R²: {best_val_r2:.4f}\n")
            f.write(f"Test R²: {test_r2:.4f}\n\n")
            f.write("Metrics by angle:\n")
            f.write("-"*30 + "\n")
            for angle, angle_metrics in metrics.items():
                f.write(f"\n{angle}:\n")
                for metric, value in angle_metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
        
        print(f"\nSummary metrics saved to: {summary_file}")
    
    print("\nTraining completed for all legs!")

if __name__ == "__main__":
    main() 