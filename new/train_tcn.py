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

class SequenceDataset(Dataset):
    """Dataset for creating sequences from time series data"""
    def __init__(self, features, targets, sequence_length, predict_start=400, predict_end=1000):
        """
        Args:
            features: Input features
            targets: Target values
            sequence_length: Length of sequences for sequential data
            predict_start: Start frame for prediction (default: 400)
            predict_end: End frame for prediction (default: 1000)
        """
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        self.predict_start = predict_start
        self.predict_end = predict_end
        
        # Calculate where we need to start collecting input data
        self.context_start = predict_start - sequence_length  # e.g., 400 - 50 = 350
        
        print(f"\nSequence Context Information:")
        print(f"Window size: {sequence_length}")
        print(f"To predict frame 400: using frames {self.context_start}-399 as input")
        print(f"To predict frame 401: using frames {self.context_start+1}-400 as input")
        print(f"And so on until frame {predict_end}")
        print(f"For each prediction, using {sequence_length} previous frames as context")
        
        # Calculate valid sequences for each trial
        self.sequences = []
        trial_length = 1400  # Full trial length
        num_trials = len(features) // trial_length
        
        print(f"\nCreating sequences for {num_trials} trials:")
        for trial in range(num_trials):
            trial_start = trial * trial_length
            trial_end = (trial + 1) * trial_length
            
            # For each frame we want to predict (400-1000)
            for pred_frame in range(predict_start, predict_end + 1):
                # Calculate absolute frame indices
                abs_pred_frame = trial_start + pred_frame
                abs_context_start = abs_pred_frame - sequence_length
                
                # Verify we have enough context AND all frames are within the same trial
                if abs_context_start >= trial_start:  # Context starts within current trial
                    self.sequences.append({
                        'context_start': abs_context_start,
                        'context_end': abs_pred_frame,  # exclusive
                        'target_frame': abs_pred_frame,
                        'trial_id': trial  # Add trial_id for verification
                    })
        
        print(f"Created {len(self.sequences)} valid sequences")
        print(f"Each sequence uses frames (t-{sequence_length}) to (t-1) to predict frame t")
        print(f"For each trial, predicting frames {predict_start}-{predict_end}")
        print(f"Total prediction frames per trial: {predict_end - predict_start + 1}")
        
        # Verify no sequences cross trial boundaries
        for seq in self.sequences:
            trial_start = seq['trial_id'] * trial_length
            trial_end = (seq['trial_id'] + 1) * trial_length
            assert seq['context_start'] >= trial_start, "Context starts before trial"
            assert seq['target_frame'] < trial_end, "Target frame beyond trial"
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Get input sequence (frames before the prediction frame)
        input_seq = self.features[seq['context_start']:seq['context_end']]
        
        # Get target (single frame we want to predict)
        target = self.targets[seq['target_frame']]
        
        # Convert to tensors and ensure correct shapes
        # Input shape: [features, sequence]
        input_tensor = torch.FloatTensor(input_seq.T)
        # Target shape: [output_features]
        target_tensor = torch.FloatTensor(target)
        
        return input_tensor, target_tensor

class ZScoreScaler:
    """Scaler for z-score normalization with per-feature parameters"""
    def __init__(self, means, stds, feature_names):
        self.means = means
        self.stds = stds
        self.feature_names = feature_names
    
    def transform(self, X):
        X_scaled = np.zeros_like(X)
        for i, feature in enumerate(self.feature_names):
            X_scaled[:, i] = (X[:, i] - self.means[feature]) / self.stds[feature]
        return X_scaled
    
    def inverse_transform(self, X):
        X_inv = np.zeros_like(X)
        for i, feature in enumerate(self.feature_names):
            X_inv[:, i] = X[:, i] * self.stds[feature] + self.means[feature]
        return X_inv

def prepare_data(config):
    """Prepare sequences of data for TCN"""
    print("\nPreparing data...")
    
    # Load and preprocess data
    df = pd.read_csv(config['data_path'])
    print(f"Loaded data with shape: {df.shape}")
    
    # Filter for specified genotype
    df = df[df['genotype'] == config['genotype']].copy()
    print(f"Filtered for {config['genotype']}: {df.shape}")
    
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
    
    # Define input features
    velocity_features = [
        'x_vel_ma5', 'y_vel_ma5', 'z_vel_ma5',
        'x_vel_ma10', 'y_vel_ma10', 'z_vel_ma10',
        'x_vel_ma20', 'y_vel_ma20', 'z_vel_ma20',
        'x_vel', 'y_vel', 'z_vel'
    ]
    
    # Add position input from config
    input_features = velocity_features + [config['input_features'][-1]]
    
    # Get output features from config
    output_features = config['output_features']
    
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
    
    # Create sequence datasets
    train_dataset = SequenceDataset(X_train_scaled, y_train_scaled, config['sequence_length'], 
                                  predict_start=400, predict_end=1000)
    val_dataset = SequenceDataset(X_val_scaled, y_val_scaled, config['sequence_length'],
                                predict_start=400, predict_end=1000)
    test_dataset = SequenceDataset(X_test_scaled, y_test_scaled, config['sequence_length'],
                                 predict_start=400, predict_end=1000)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    return (train_loader, val_loader, test_loader), (X_scaler, y_scaler), (input_features, output_features)

def train_model(model, train_loader, val_loader, config):
    """Train the TCN model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\nGPU Memory after clearing cache:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    criterion = nn.L1Loss()  # MAE loss
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Train]')
        
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
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Val]')
        
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
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    return best_model, best_val_loss

def evaluate_model(model, test_loader, y_scaler, output_features, plots_dir, config, X_scaler):
    """Evaluate the trained TCN model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Get data from the original dataset
    df = pd.read_csv(config['data_path'])
    df = df[df['genotype'] == config['genotype']]  # Use genotype from config
    print(f"Evaluating {config['genotype']} data")
    
    # Split data into fixed-size trials of 1400 frames
    trial_size = 1400
    num_trials = len(df) // trial_size
    
    # Create trial indices and shuffle with same seed as in prepare_data
    trial_indices = np.arange(num_trials)
    np.random.seed(42)  # Same seed as in prepare_data
    np.random.shuffle(trial_indices)
    
    # Get test indices (last 15% of trials)
    train_size = int(0.7 * num_trials)
    val_size = int(0.15 * num_trials)
    test_indices = trial_indices[train_size + val_size:]
    
    print(f"\nEvaluating {len(test_indices)} test trials...")
    
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
    
    # Create a dictionary to store data by trial
    trial_data = {}
    for trial_id in test_indices:  # Only process test trials
        start_idx = trial_id * trial_size
        end_idx = (trial_id + 1) * trial_size
        trial_df = df.iloc[start_idx:end_idx]
        
        # Get features and targets for this trial
        X = trial_df[config['input_features']].values
        y = trial_df[output_features].values
        
        # Scale the data
        X_scaled = X_scaler.transform(X)
        y_scaled = y_scaler.transform(y)
        
        # Store original and scaled data
        trial_data[trial_id] = {
            'X': X_scaled,
            'y': y_scaled,
            'original_y': y
        }
    
    # Calculate mean and std for each angle across all test trials
    angle_stats = {}
    for i, angle in enumerate(output_features):
        all_values = []
        for trial_id in trial_data:
            all_values.extend(trial_data[trial_id]['original_y'][:, i])
        angle_stats[angle] = {
            'mean': np.mean(all_values),
            'std': np.std(all_values)
        }
    
    # Process each test trial
    metrics = {}
    # Dictionary to store figures for each angle
    angle_figures = {angle: [] for angle in output_features}
    
    for trial_id in trial_data:
        print(f"\nProcessing test trial {trial_id}...")
        X = trial_data[trial_id]['X']
        original_y = trial_data[trial_id]['original_y']
        
        predictions = []
        # Generate predictions for frames 400-1000
        for frame in range(400, 1001):
            # Get the sequence window
            start_frame = frame - config['sequence_length']
            window = X[start_frame:frame]
            
            # Skip if we don't have enough context
            if len(window) < config['sequence_length']:
                continue
                
            # Prepare input tensor
            window_tensor = torch.FloatTensor(window.T).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                pred = model(window_tensor)
                pred = pred.cpu().numpy()
                # Reshape to 2D array (1 sample, n_features)
                pred = pred.reshape(1, -1)
                # Inverse transform
                pred = y_scaler.inverse_transform(pred)
                predictions.append(pred[0])
        
        predictions = np.array(predictions)
        
        # Calculate metrics for each angle
        for i, angle in enumerate(output_features):
            if angle not in metrics:
                metrics[angle] = {
                    'mae_raw': [], 'rmse_raw': [], 'r2_raw': [],
                    'mae_z': [], 'rmse_z': [], 'r2_z': [],
                    'mean': angle_stats[angle]['mean'],
                    'std': angle_stats[angle]['std']
                }
            
            # Get true values for frames 400-1000
            true = original_y[400:1001, i]
            pred = predictions[:, i]
            
            # Ensure same length for comparison
            min_len = min(len(pred), len(true))
            pred = pred[:min_len]
            true = true[:min_len]
            
            # Calculate raw metrics
            mae_raw = np.mean(np.abs(pred - true))
            rmse_raw = np.sqrt(np.mean((pred - true)**2))
            r2_raw = 1 - np.sum((true - pred)**2) / np.sum((true - true.mean())**2)
            
            # Calculate z-scored metrics
            mean = angle_stats[angle]['mean']
            std = angle_stats[angle]['std']
            true_z = (true - mean) / std
            pred_z = (pred - mean) / std
            
            mae_z = np.mean(np.abs(pred_z - true_z))
            rmse_z = np.sqrt(np.mean((pred_z - true_z)**2))
            r2_z = 1 - np.sum((true_z - pred_z)**2) / np.sum((true_z - true_z.mean())**2)
            
            # Store both raw and z-scored metrics
            metrics[angle]['mae_raw'].append(mae_raw)
            metrics[angle]['rmse_raw'].append(rmse_raw)
            metrics[angle]['r2_raw'].append(r2_raw)
            metrics[angle]['mae_z'].append(mae_z)
            metrics[angle]['rmse_z'].append(rmse_z)
            metrics[angle]['r2_z'].append(r2_z)
            
            # Create figure for this trial and angle
            fig = plt.figure(figsize=(12, 6))
            
            # Plot full sequence for context
            time_steps = range(len(original_y[:, i]))
            plt.plot(time_steps, original_y[:, i], label='Actual', alpha=0.7)
            
            # Plot predictions
            pred_time_steps = range(400, 400 + len(pred))
            plt.plot(pred_time_steps, pred, label='Predicted', alpha=0.7, color='red')
            
            # Add vertical line at frame 400
            plt.axvline(x=400, color='k', linestyle='--', alpha=0.5, label='Frame 400')
            
            plt.title(f'Test Trial {trial_id} - {angle} Predictions\nRaw MAE = {mae_raw:.3f}°, Z-scored MAE = {mae_z:.3f}')
            plt.xlabel('Frame Number')
            plt.ylabel('Angle (degrees)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add metrics text box
            plt.text(0.02, 0.98, 
                    f'Raw Metrics:\n'
                    f'  MAE: {mae_raw:.3f}°\n'
                    f'  RMSE: {rmse_raw:.3f}°\n'
                    f'  R²: {r2_raw:.3f}\n'
                    f'Z-scored Metrics:\n'
                    f'  MAE: {mae_z:.3f}\n'
                    f'  RMSE: {rmse_z:.3f}\n'
                    f'  R²: {r2_z:.3f}\n'
                    f'Statistics:\n'
                    f'  Mean: {mean:.1f}°\n'
                    f'  Std: {std:.1f}°',
                    transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', alpha=0.8),
                    verticalalignment='top')
            
            # Store figure for combined PDF
            angle_figures[angle].append(fig)
            plt.close()
    
    # Create combined PDFs for each angle
    from matplotlib.backends.backend_pdf import PdfPages
    
    print("\nCreating combined PDFs for each angle...")
    for angle, figures in angle_figures.items():
        pdf_path = plots_dir / f'{angle}_test_trials.pdf'
        with PdfPages(pdf_path) as pdf:
            for fig in figures:
                pdf.savefig(fig)
                plt.close(fig)
        print(f"Created {pdf_path}")
    
    # Calculate and plot overall metrics
    final_metrics = {}
    for angle in output_features:
        final_metrics[angle] = {
            'mae_raw': np.mean(metrics[angle]['mae_raw']),
            'rmse_raw': np.mean(metrics[angle]['rmse_raw']),
            'r2_raw': np.mean(metrics[angle]['r2_raw']),
            'mae_z': np.mean(metrics[angle]['mae_z']),
            'rmse_z': np.mean(metrics[angle]['rmse_z']),
            'r2_z': np.mean(metrics[angle]['r2_z']),
            'mean': metrics[angle]['mean'],
            'std': metrics[angle]['std']
        }
    
    # Print summary of metrics
    print("\nMetrics Summary (Averaged across test trials):")
    print("\nRaw Metrics:")
    maes_raw = [m['mae_raw'] for m in final_metrics.values()]
    print(f"Average MAE across all angles: {np.mean(maes_raw):.3f}°")
    print(f"Best MAE: {min(maes_raw):.3f}°")
    print(f"Worst MAE: {max(maes_raw):.3f}°")
    
    print("\nZ-scored Metrics:")
    maes_z = [m['mae_z'] for m in final_metrics.values()]
    print(f"Average MAE across all angles: {np.mean(maes_z):.3f}")
    print(f"Best MAE: {min(maes_z):.3f}")
    print(f"Worst MAE: {max(maes_z):.3f}")
    
    print("\nAngle Statistics:")
    for angle, stats in final_metrics.items():
        print(f"{angle}:")
        print(f"  Mean: {stats['mean']:.1f}°")
        print(f"  Std: {stats['std']:.1f}°")
    
    return final_metrics

def run_experiment(position_input, leg_dir, config):
    """Run a single experiment with a specific position input"""
    print(f"\n{'='*80}")
    print(f"Running experiment with position input: {position_input}")
    print(f"{'='*80}")
    
    # Create experiment-specific directories
    exp_name = f"tcn_{position_input.replace('-', '_')}"
    results_dir = leg_dir / exp_name  # Now using leg_dir as parent
    models_dir = results_dir / 'models'
    plots_dir = results_dir / 'plots'
    
    # Clean up old results if they exist
    if results_dir.exists():
        shutil.rmtree(results_dir)
    
    results_dir.mkdir(exist_ok=True, parents=True)
    models_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    
    print("\nCreated directory structure:")
    print(f"- {results_dir}")
    print(f"- {models_dir}")
    print(f"- {plots_dir}")
    
    # Update config with position input
    config['input_features'] = [
        'x_vel_ma5', 'y_vel_ma5', 'z_vel_ma5',
        'x_vel_ma10', 'y_vel_ma10', 'z_vel_ma10',
        'x_vel_ma20', 'y_vel_ma20', 'z_vel_ma20',
        'x_vel', 'y_vel', 'z_vel',
        position_input
    ]
    
    # Use prepare_data to get consistent data preprocessing
    (train_loader, val_loader, test_loader), (X_scaler, y_scaler), (input_features, output_features) = prepare_data(config)
    
    # Create model
    model = TemporalConvNet(
        input_size=len(input_features),
        output_size=len(output_features),
        num_channels=config['num_channels'],
        kernel_size=config['kernel_size'],
        dropout=config['dropout'],  # Added dropout parameter
        sequence_length=config['sequence_length']
    )
    
    # Train model
    best_model_state, best_val_loss = train_model(model, train_loader, val_loader, config)
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Save configuration for evaluation
    config['input_features'] = input_features
    config['output_features'] = output_features
    
    # Evaluate model
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        y_scaler=y_scaler,
        output_features=output_features,
        plots_dir=plots_dir,
        config=config,
        X_scaler=X_scaler
    )
    
    # Save model and results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_save_path = models_dir / f'model_{timestamp}.pth'
    
    results = {
        'model_state_dict': best_model_state,
        'config': config,
        'input_features': input_features,
        'output_features': output_features,
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'metrics': metrics,
        'best_val_loss': best_val_loss
    }
    
    torch.save(results, model_save_path)
    print(f"\nModel and results saved to: {model_save_path}")
    
    return metrics, best_val_loss

def objective(trial, leg_prefix, leg_angles, position_input, base_data_path):
    """Optuna objective function for hyperparameter optimization"""
    # Get genotype from leg_prefix (format: "genotype_leg")
    genotype = leg_prefix.split('_')[0]
    
    # Use fixed parameters for first trial, random for subsequent trials
    if trial.number == 0:
        # Fixed parameters for first trial
        config = {
            'data_path': base_data_path,
            'batch_size': 64,
            'learning_rate': 1e-3,
            'num_epochs': 400,
            'patience': 20,
            'sequence_length': 50,
            'kernel_size': 3,
            'dropout': 0.2,
            'output_features': leg_angles,
            'genotype': genotype,
            'num_channels': [64, 128, 256]  # Fixed channel sizes
        }
        print("\nUsing fixed parameters for first trial:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    else:
        # Random parameters for subsequent trials
        config = {
            'data_path': base_data_path,
            'batch_size': trial.suggest_int('batch_size', 32, 128, step=32),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'num_epochs': 400,
            'patience': 20,
            'sequence_length': trial.suggest_int('sequence_length', 30, 200, step=10),
            'kernel_size': trial.suggest_int('kernel_size', 2, 5),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'output_features': leg_angles,
            'genotype': genotype
        }
        
        # Random number of layers and channels for subsequent trials
        n_layers = trial.suggest_int('n_layers', 2, 4)
        channels = []
        for i in range(n_layers):
            if i == 0:
                channels.append(trial.suggest_int(f'channels_{i}', 32, 128, step=32))
            else:
                channels.append(trial.suggest_int(f'channels_{i}', channels[-1], channels[-1]*2))
        config['num_channels'] = channels
        
        print(f"\nUsing random parameters for trial {trial.number}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    # Update input features
    config['input_features'] = [
        'x_vel_ma5', 'y_vel_ma5', 'z_vel_ma5',
        'x_vel_ma10', 'y_vel_ma10', 'z_vel_ma10',
        'x_vel_ma20', 'y_vel_ma20', 'z_vel_ma20',
        'x_vel', 'y_vel', 'z_vel',
        position_input
    ]
    
    try:
        # Prepare data
        (train_loader, val_loader, test_loader), (X_scaler, y_scaler), (input_features, output_features) = prepare_data(config)
        
        # Create model
        model = TemporalConvNet(
            input_size=len(input_features),
            output_size=len(output_features),
            num_channels=config['num_channels'],
            kernel_size=config['kernel_size'],
            dropout=config['dropout'],
            sequence_length=config['sequence_length']
        )
        
        # Train model
        best_model_state, best_val_loss = train_model(model, train_loader, val_loader, config)
        
        return best_val_loss
        
    except Exception as e:
        print(f"Trial failed: {e}")
        raise optuna.exceptions.TrialPruned()

def optimize_hyperparameters(leg_prefix, leg_angles, position_input, base_data_path, n_trials=15):
    """Run hyperparameter optimization"""
    if n_trials == 1:
        # Return fixed parameters directly
        return {
            'batch_size': 64,
            'learning_rate': 1e-3,
            'sequence_length': 50,
            'kernel_size': 3,
            'dropout': 0.2,
            'n_layers': 3,
            'channels_0': 64,
            'channels_1': 128,
            'channels_2': 256
        }
    
    # Run optimization for multiple trials
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, leg_prefix, leg_angles, position_input, base_data_path),
        n_trials=n_trials
    )
    
    print("\nBest trial:")
    trial = study.best_trial
    
    print(f"  Value: {trial.value}")
    print("\nBest hyperparameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    
    return trial.params

def main():
    """Main function to run all experiments"""
    # Check GPU availability and print info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\nDevice Information:")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    base_data_path = "Z:/Divya/TEMP_transfers/toAni/BPN_P9LT_P9RT_flyCoords.csv"
    
    # Define genotypes to process
    genotypes = ['P9LT', 'P9RT', 'BPN']
    
    # Define legs and their corresponding features
    legs = {
        'R-F': ['R1A_flex', 'R1A_rot', 'R1A_abduct', 'R1B_flex', 'R1B_rot', 'R1C_flex', 'R1C_rot', 'R1D_flex'],
        'L-F': ['L1A_flex', 'L1A_rot', 'L1A_abduct', 'L1B_flex', 'L1B_rot', 'L1C_flex', 'L1C_rot', 'L1D_flex'],
        'R-M': ['R2A_flex', 'R2A_rot', 'R2A_abduct', 'R2B_flex', 'R2B_rot', 'R2C_flex', 'R2C_rot', 'R2D_flex'],
        'L-M': ['L2A_flex', 'L2A_rot', 'L2A_abduct', 'L2B_flex', 'L2B_rot', 'L2C_flex', 'L2C_rot', 'L2D_flex'],
        'R-H': ['R3A_flex', 'R3A_rot', 'R3A_abduct', 'R3B_flex', 'R3B_rot', 'R3C_flex', 'R3C_rot', 'R3D_flex'],
        'L-H': ['L3A_flex', 'L3A_rot', 'L3A_abduct', 'L3B_flex', 'L3B_rot', 'L3C_flex', 'L3C_rot', 'L3D_flex']
    }
    
    # Joint positions to test for each leg
    joint_positions = ['ThC', 'CTr', 'FeTi', 'TiTa', 'TaG']
    coordinates = ['x', 'y', 'z']
    
    # Create base results directory
    base_results_dir = Path('tcn_experiments')
    base_results_dir.mkdir(exist_ok=True)
    
    # Process each genotype
    for genotype in genotypes:
        print(f"\n{'#'*100}")
        print(f"Processing genotype: {genotype}")
        print(f"{'#'*100}")
        
        # Create genotype-specific directory
        genotype_dir = base_results_dir / genotype
        genotype_dir.mkdir(exist_ok=True)
        
        # Run experiments for each leg
        for leg_prefix, leg_angles in legs.items():
            print(f"\n{'='*80}")
            print(f"Processing {leg_prefix} leg")
            print(f"{'='*80}")
            
            # Create leg-specific directory
            leg_dir = genotype_dir / f"{leg_prefix}_leg"
            leg_dir.mkdir(exist_ok=True)
            
            # Generate position inputs for this leg
            position_inputs = [
                f'{leg_prefix}-{joint}_{coord}'
                for joint in joint_positions
                for coord in coordinates
            ]
            
            print(f"\nTesting {len(position_inputs)} position inputs:")
            for pos in position_inputs:
                print(f"  - {pos}")
            
            # Store results for comparison
            experiment_results = {}
            
            # Run experiments for each position input
            for position_input in position_inputs:
                print(f"\nTesting position input: {position_input}")
                
                # First, optimize hyperparameters
                print("\nOptimizing hyperparameters...")
                best_params = optimize_hyperparameters(
                    leg_prefix=f"{genotype}_{leg_prefix}",
                    leg_angles=leg_angles,
                    position_input=position_input,
                    base_data_path=base_data_path,
                    n_trials=1
                )
                
                # Create config with best parameters
                config = {
                    'data_path': base_data_path,
                    'batch_size': best_params['batch_size'],
                    'learning_rate': best_params['learning_rate'],
                    'num_epochs': 200,
                    'patience': 20,
                    'sequence_length': best_params['sequence_length'],
                    'kernel_size': best_params['kernel_size'],
                    'num_channels': [best_params[f'channels_{i}'] for i in range(best_params['n_layers'])],
                    'dropout': best_params['dropout'],
                    'output_features': leg_angles,
                    'genotype': genotype
                }
                
                # Run experiment with optimized hyperparameters
                metrics, val_loss = run_experiment(position_input, leg_dir, config)
                
                experiment_results[position_input] = {
                    'metrics': metrics,
                    'val_loss': val_loss,
                    'best_params': best_params
                }
            
            # Save comparison results for this leg
            comparison_path = leg_dir / 'experiment_comparison.json'
            with open(comparison_path, 'w') as f:
                json.dump(experiment_results, f, indent=2)
            
            # Create comparison plots for this leg
            create_comparison_plots(experiment_results, position_inputs, leg_dir)
        
        # Create genotype summary
        print(f"\nCreating summary for {genotype}...")
        create_genotype_summary(genotype_dir)
    
    print("\nAll experiments completed!")

def create_genotype_summary(genotype_dir):
    """Create summary plots and metrics for a genotype"""
    # Load results from all legs
    leg_results = {}
    for leg_dir in genotype_dir.glob('*_leg'):
        leg_prefix = leg_dir.name.replace('_leg', '')
        with open(leg_dir / 'experiment_comparison.json', 'r') as f:
            leg_results[leg_prefix] = json.load(f)
    
    # Create summary plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))
    
    # Collect best positions for each leg
    best_positions = {}
    for leg_prefix, results in leg_results.items():
        # Find best position based on average z-scored MAE
        position_scores = {
            pos: np.mean([m['mae_z'] for m in metrics['metrics'].values()])
            for pos, metrics in results.items()
        }
        best_pos = min(position_scores.items(), key=lambda x: x[1])
        best_positions[leg_prefix] = best_pos
    
    # Plot best positions
    legs = list(best_positions.keys())
    x_pos = np.arange(len(legs))
    
    # Z-scored plot
    scores_z = [pos[1] for pos in best_positions.values()]
    ax1.bar(x_pos, scores_z)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(legs, rotation=45, ha='right')
    ax1.set_title('Best Position Z-scored MAE by Leg')
    ax1.set_xlabel('Leg')
    ax1.set_ylabel('Z-scored MAE')
    
    # Add best position labels
    for i, (leg, (pos, score)) in enumerate(best_positions.items()):
        ax1.text(i, score, pos, rotation=45, ha='left', va='bottom')
    
    # Raw MAE plot
    scores_raw = [
        np.mean([m['mae_raw'] for m in leg_results[leg][pos[0]]['metrics'].values()])
        for leg, pos in best_positions.items()
    ]
    ax2.bar(x_pos, scores_raw)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(legs, rotation=45, ha='right')
    ax2.set_title('Best Position Raw MAE by Leg')
    ax2.set_xlabel('Leg')
    ax2.set_ylabel('MAE (degrees)')
    
    plt.tight_layout()
    plt.savefig(genotype_dir / 'leg_comparison.png')
    plt.close()
    
    # Save summary to JSON
    summary = {
        'best_positions': {leg: {'position': pos[0], 'score': pos[1]} 
                          for leg, pos in best_positions.items()},
        'average_scores': {
            'z_scored_mae': np.mean(scores_z),
            'raw_mae': np.mean(scores_raw)
        }
    }
    
    with open(genotype_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

def create_comparison_plots(experiment_results, position_inputs, output_dir):
    """Create comparison plots for a leg's experiments"""
    # Sort results by z-scored MAE
    sorted_results_z = []
    sorted_results_raw = []
    for position_input in position_inputs:
        results = experiment_results[position_input]
        # Z-scored MAE
        avg_mae_z = np.mean([m['mae_z'] for m in results['metrics'].values()])
        sorted_results_z.append((position_input, avg_mae_z, results))
        # Raw MAE
        avg_mae_raw = np.mean([m['mae_raw'] for m in results['metrics'].values()])
        sorted_results_raw.append((position_input, avg_mae_raw, results))
    
    sorted_results_z.sort(key=lambda x: x[1])
    sorted_results_raw.sort(key=lambda x: x[1])
    
    # Create side-by-side comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Z-scored plot
    positions_z = [r[0] for r in sorted_results_z]
    maes_z = [r[1] for r in sorted_results_z]
    x_positions = np.arange(len(positions_z))
    ax1.bar(x_positions, maes_z)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(positions_z, rotation=45, ha='right')
    ax1.set_title('Average Z-scored MAE by Position Input')
    ax1.set_xlabel('Position Input')
    ax1.set_ylabel('Z-scored MAE')
    ax1.grid(True, alpha=0.3)
    
    # Raw plot
    positions_raw = [r[0] for r in sorted_results_raw]
    maes_raw = [r[1] for r in sorted_results_raw]
    x_positions = np.arange(len(positions_raw))
    ax2.bar(x_positions, maes_raw)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(positions_raw, rotation=45, ha='right')
    ax2.set_title('Average Raw MAE by Position Input')
    ax2.set_xlabel('Position Input')
    ax2.set_ylabel('MAE (degrees)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'position_comparison.png')
    plt.close()

if __name__ == "__main__":
    main() 