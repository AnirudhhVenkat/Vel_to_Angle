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

class Chomp1d(nn.Module):
    """Helper module to ensure causal convolutions"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    """Temporal block with dilated convolutions"""
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
        """Initialize weights using Xavier initialization"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network"""
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.2):
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
        self.linear = nn.Linear(num_channels[-1], output_size)
        
        print("\nModel Architecture:")
        print(f"Input size: {input_size}")
        print(f"Hidden channels: {num_channels}")
        print(f"Output size: {output_size}")
        print(f"Kernel size: {kernel_size}")
        print(f"Dropout: {dropout}")
        
    def forward(self, x):
        # Input shape: [batch, features, sequence]
        y = self.network(x)
        # Take the last time step
        y = y[:, :, -1]
        # Project to output size
        y = self.linear(y)
        return y

class ZScoreScaler:
    """Custom scaler for z-score normalization with feature names"""
    def __init__(self, means, stds, feature_names):
        self.means = means
        self.stds = stds
        self.feature_names = feature_names
    
    def transform(self, X):
        """Z-score normalize the input data"""
        X_scaled = np.zeros_like(X)
        for i, feature in enumerate(self.feature_names):
            X_scaled[:, i] = (X[:, i] - self.means[feature]) / self.stds[feature]
        return X_scaled
    
    def inverse_transform(self, X):
        """Inverse transform z-scored data back to original scale"""
        X_orig = np.zeros_like(X)
        for i, feature in enumerate(self.feature_names):
            X_orig[:, i] = X[:, i] * self.stds[feature] + self.means[feature]
        return X_orig

def prepare_data(config):
    """Prepare data for TCN without windowing"""
    print("\nPreparing data...")
    
    # Load and preprocess data
    df = pd.read_csv(config['data_path'])
    print(f"Loaded data with shape: {df.shape}")
    
    # Filter for specified genotype if not ALL
    if config['genotype'] != 'ALL':
        df = df[df['genotype'] == config['genotype']].copy()
        print(f"Filtered for {config['genotype']}: {df.shape}")
    
    # Get trial indices (assuming 1400 frames per trial)
    trial_size = 1400
    num_trials = len(df) // trial_size
    print(f"\nTotal number of trials: {num_trials}")
    
    # Filter frames 200-1000 from each trial
    filtered_df = pd.DataFrame()
    for trial in range(num_trials):
        start_idx = trial * trial_size + 200  # Start at frame 200 instead of 400
        end_idx = trial * trial_size + 1000   # End at frame 1000
        trial_data = df.iloc[start_idx:end_idx].copy()
        filtered_df = pd.concat([filtered_df, trial_data], ignore_index=True)
    
    df = filtered_df
    print(f"Filtered to frames 200-1000 for each trial")
    print(f"New shape: {df.shape}")
    
    # Calculate moving averages for velocities within each trial
    base_velocities = ['x_vel', 'y_vel', 'z_vel']
    for window in [5, 10, 20]:
        for vel in base_velocities:
            # Calculate moving average for each trial separately
            ma_values = []
            frames_per_trial = 800  # After filtering (1000 - 200 = 800)
            for trial in range(num_trials):
                start_idx = trial * frames_per_trial
                end_idx = (trial + 1) * frames_per_trial
                trial_data = df[vel].iloc[start_idx:end_idx]
                # Calculate moving average and handle edges
                ma = trial_data.rolling(window=window, center=True, min_periods=1).mean()
                ma_values.extend(ma.tolist())
            df[f'{vel}_ma{window}'] = ma_values
    
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
    
    # Output features: all joint angles for all legs
    output_features = []
    
    # Right Front Leg (R1)
    output_features.extend(['R1A_flex', 'R1A_rot', 'R1A_abduct',
                          'R1B_flex', 'R1B_rot',
                          'R1C_flex', 'R1C_rot',
                          'R1D_flex'])
    
    # Left Front Leg (L1)
    output_features.extend(['L1A_flex', 'L1A_rot', 'L1A_abduct',
                          'L1B_flex', 'L1B_rot',
                          'L1C_flex', 'L1C_rot',
                          'L1D_flex'])
    
    # Right Middle Leg (R2)
    output_features.extend(['R2A_flex', 'R2A_rot', 'R2A_abduct',
                          'R2B_flex', 'R2B_rot',
                          'R2C_flex', 'R2C_rot',
                          'R2D_flex'])
    
    # Left Middle Leg (L2)
    output_features.extend(['L2A_flex', 'L2A_rot', 'L2A_abduct',
                          'L2B_flex', 'L2B_rot',
                          'L2C_flex', 'L2C_rot',
                          'L2D_flex'])
    
    # Right Hind Leg (R3)
    output_features.extend(['R3A_flex', 'R3A_rot', 'R3A_abduct',
                          'R3B_flex', 'R3B_rot',
                          'R3C_flex', 'R3C_rot',
                          'R3D_flex'])
    
    # Left Hind Leg (L3)
    output_features.extend(['L3A_flex', 'L3A_rot', 'L3A_abduct',
                          'L3B_flex', 'L3B_rot',
                          'L3C_flex', 'L3C_rot',
                          'L3D_flex'])
    
    print(f"\nFeature Information:")
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
    frames_per_trial = 800  # After filtering (1000 - 200 = 800)
    X_trials = X.reshape(num_trials, frames_per_trial, -1)
    y_trials = y.reshape(num_trials, frames_per_trial, -1)
    
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
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled)),
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val_scaled)),
        batch_size=config['batch_size']
    )
    
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test_scaled)),
        batch_size=config['batch_size']
    )
    
    return (train_loader, val_loader, test_loader), (X_scaler, y_scaler), (input_features, output_features)

def train_model(model, train_loader, val_loader, config):
    """Train the TCN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Use MAE (L1) loss instead of MSE
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
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
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Reshape inputs for TCN [batch, features, sequence]
            inputs = inputs.unsqueeze(-1)  # Add sequence dimension
            
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
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Reshape inputs for TCN [batch, features, sequence]
                inputs = inputs.unsqueeze(-1)  # Add sequence dimension
                
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

def evaluate_model(model, test_loader, y_scaler, output_features, plots_dir):
    """Evaluate the trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_predictions_zscored = []
    all_targets_zscored = []
    
    # Use MAE loss for evaluation
    criterion = nn.L1Loss(reduction='none')
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)  # These are already z-scored
            
            # Reshape inputs for TCN [batch, features, sequence]
            inputs = inputs.unsqueeze(-1)  # Add sequence dimension
            
            outputs = model(inputs)  # Model outputs z-scored predictions
            
            all_predictions_zscored.append(outputs.cpu().numpy())
            all_targets_zscored.append(targets.cpu().numpy())
            
            # Convert to original scale for interpretable metrics
            outputs_orig = torch.tensor(y_scaler.inverse_transform(outputs.cpu().numpy()))
            targets_orig = torch.tensor(y_scaler.inverse_transform(targets.cpu().numpy()))
            
            all_predictions.append(outputs_orig.numpy())
            all_targets.append(targets_orig.numpy())
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    predictions_zscored = np.concatenate(all_predictions_zscored)
    targets_zscored = np.concatenate(all_targets_zscored)
    
    # Calculate metrics for each angle (in original scale)
    metrics = {}
    zscored_metrics = {}
    
    for i, angle in enumerate(output_features):
        # Original scale metrics
        mae = np.mean(np.abs(predictions[:, i] - targets[:, i]))
        rmse = np.sqrt(np.mean((predictions[:, i] - targets[:, i])**2))
        r2 = 1 - np.sum((targets[:, i] - predictions[:, i])**2) / np.sum((targets[:, i] - targets[:, i].mean())**2)
        
        metrics[angle] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        # Z-scored metrics
        mae_z = np.mean(np.abs(predictions_zscored[:, i] - targets_zscored[:, i]))
        rmse_z = np.sqrt(np.mean((predictions_zscored[:, i] - targets_zscored[:, i])**2))
        r2_z = 1 - np.sum((targets_zscored[:, i] - predictions_zscored[:, i])**2) / np.sum((targets_zscored[:, i] - targets_zscored[:, i].mean())**2)
        
        zscored_metrics[angle] = {
            'mae': mae_z,
            'rmse': rmse_z,
            'r2': r2_z
        }
    
    # Create bar plots of z-scored MAE by leg
    legs = {
        'R1': ['R1A_flex', 'R1A_rot', 'R1A_abduct', 'R1B_flex', 'R1B_rot', 'R1C_flex', 'R1C_rot', 'R1D_flex'],
        'L1': ['L1A_flex', 'L1A_rot', 'L1A_abduct', 'L1B_flex', 'L1B_rot', 'L1C_flex', 'L1C_rot', 'L1D_flex'],
        'R2': ['R2A_flex', 'R2A_rot', 'R2A_abduct', 'R2B_flex', 'R2B_rot', 'R2C_flex', 'R2C_rot', 'R2D_flex'],
        'L2': ['L2A_flex', 'L2A_rot', 'L2A_abduct', 'L2B_flex', 'L2B_rot', 'L2C_flex', 'L2C_rot', 'L2D_flex'],
        'R3': ['R3A_flex', 'R3A_rot', 'R3A_abduct', 'R3B_flex', 'R3B_rot', 'R3C_flex', 'R3C_rot', 'R3D_flex'],
        'L3': ['L3A_flex', 'L3A_rot', 'L3A_abduct', 'L3B_flex', 'L3B_rot', 'L3C_flex', 'L3C_rot', 'L3D_flex']
    }
    
    # Create bar plot of z-scored MAE for all angles
    plt.figure(figsize=(20, 10))
    angles = list(zscored_metrics.keys())
    mae_values = [zscored_metrics[angle]['mae'] for angle in angles]
    
    # Create bars with different colors for each leg
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#99ccff']
    bar_colors = []
    for angle in angles:
        for i, (leg, leg_angles) in enumerate(legs.items()):
            if angle in leg_angles:
                bar_colors.append(colors[i])
                break
    
    bars = plt.bar(range(len(angles)), mae_values, color=bar_colors)
    
    # Customize the plot
    plt.title('Z-Scored MAE by Joint Angle', fontsize=14, pad=20)
    plt.xlabel('Joint Angles', fontsize=12)
    plt.ylabel('Z-Scored MAE', fontsize=12)
    plt.xticks(range(len(angles)), angles, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=leg) 
                      for leg, color in zip(legs.keys(), colors)]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(plots_dir / 'zscored_mae_all_angles.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create separate bar plots for each leg
    for leg_name, leg_angles in legs.items():
        plt.figure(figsize=(15, 8))
        
        # Get MAE values for this leg's angles
        leg_mae_values = [zscored_metrics[angle]['mae'] for angle in leg_angles]
        
        # Create bars
        bars = plt.bar(range(len(leg_angles)), leg_mae_values, color=colors[list(legs.keys()).index(leg_name)])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        # Customize the plot
        plt.title(f'{leg_name} Z-Scored MAE by Joint Angle', fontsize=14, pad=20)
        plt.xlabel('Joint Angles', fontsize=12)
        plt.ylabel('Z-Scored MAE', fontsize=12)
        plt.xticks(range(len(leg_angles)), leg_angles, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(plots_dir / f'zscored_mae_{leg_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create a PDF for each leg with both original and z-scored metrics
    for leg_name, leg_angles in legs.items():
        pdf_path = plots_dir / f'predictions_{leg_name}.pdf'
        with PdfPages(pdf_path) as pdf:
            # First page: Summary metrics for this leg
            plt.figure(figsize=(12, 8))
            plt.axis('off')
            plt.text(0.1, 0.95, f'{leg_name} Summary Metrics', fontsize=16, fontweight='bold')
            
            y_pos = 0.85
            for angle_name in leg_angles:
                orig_metrics = metrics[angle_name]
                z_metrics = zscored_metrics[angle_name]
                
                plt.text(0.1, y_pos, f"\n{angle_name}:", fontsize=12, fontweight='bold')
                y_pos -= 0.05
                plt.text(0.1, y_pos, f"  Original Scale:")
                y_pos -= 0.03
                plt.text(0.1, y_pos, f"    MAE: {orig_metrics['mae']:.4f}°")
                y_pos -= 0.03
                plt.text(0.1, y_pos, f"    RMSE: {orig_metrics['rmse']:.4f}°")
                y_pos -= 0.03
                plt.text(0.1, y_pos, f"    R²: {orig_metrics['r2']:.4f}")
                y_pos -= 0.03
                plt.text(0.1, y_pos, f"  Z-scored Scale:")
                y_pos -= 0.03
                plt.text(0.1, y_pos, f"    MAE: {z_metrics['mae']:.4f}")
                y_pos -= 0.03
                plt.text(0.1, y_pos, f"    RMSE: {z_metrics['rmse']:.4f}")
                y_pos -= 0.03
                plt.text(0.1, y_pos, f"    R²: {z_metrics['r2']:.4f}")
                y_pos -= 0.05
            
            pdf.savefig()
            plt.close()
            
            # Plot each trial for this leg
            frames_per_trial = 800  # After filtering (1000 - 200 = 800)
            num_trials = len(predictions) // frames_per_trial
            predictions_by_trial = predictions.reshape(num_trials, frames_per_trial, -1)
            targets_by_trial = targets.reshape(num_trials, frames_per_trial, -1)
            
            for trial in range(num_trials):
                # Create a wider figure
                fig = plt.figure(figsize=(24, 12))
                plt.suptitle(f'{leg_name} - Trial {trial + 1} of {num_trials}', fontsize=16, y=0.95)
                
                # Get indices of angles for this leg
                angle_indices = [output_features.index(angle) for angle in leg_angles]
                
                # Create subplots for each angle
                for i, (angle_idx, angle_name) in enumerate(zip(angle_indices, leg_angles)):
                    plt.subplot(2, 4, i + 1)  # 2 rows, 4 columns for 8 angles
                    
                    # Get predictions and targets for this angle
                    pred_trial = predictions_by_trial[trial, :, angle_idx]
                    target_trial = targets_by_trial[trial, :, angle_idx]
                    
                    # Plot actual values
                    plt.plot(target_trial, 'b-', label='Actual', alpha=0.7, linewidth=2)
                    
                    # Plot predicted values
                    plt.plot(pred_trial, 'r-', label='Predicted', alpha=0.7, linewidth=2)
                    
                    # Calculate trial-specific metrics
                    mae = np.mean(np.abs(pred_trial - target_trial))
                    r2 = 1 - np.sum((target_trial - pred_trial)**2) / \
                         np.sum((target_trial - target_trial.mean())**2)
                    
                    plt.title(f"{angle_name}\nMAE: {mae:.3f}°, R²: {r2:.3f}", fontsize=12)
                    plt.xlabel('Frame', fontsize=10)
                    plt.ylabel('Angle (°)', fontsize=10)
                    plt.legend(fontsize=10)
                    plt.grid(True, alpha=0.3)
                    plt.tick_params(labelsize=10)
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                pdf.savefig(fig)
                plt.close()
        
        print(f"\nPrediction plots for {leg_name} saved to: {pdf_path}")
    
    return metrics, zscored_metrics

def main():
    """Main function to train and evaluate the TCN model."""
    # Configuration
    config = {
        'data_path': "Z:/Divya/TEMP_transfers/toAni/BPN_P9LT_P9RT_flyCoords.csv",
        'genotype': 'ALL',  # Use all genotypes
        'batch_size': 64,
        'learning_rate': 1e-4,
        'num_epochs': 200,
        'patience': 20,
        'num_channels': [64, 128, 256],  # Number of channels in each layer
        'kernel_size': 3,
        'dropout': 0.2
    }
    
    # Create directories
    results_dir = Path('tcn_nowindow_results')
    models_dir = results_dir / 'models'
    plots_dir = results_dir / 'plots'
    
    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    
    print("\nCreated directory structure:")
    print(f"- {results_dir}")
    print(f"- {models_dir}")
    print(f"- {plots_dir}")
    
    # Prepare data
    (train_loader, val_loader, test_loader), (X_scaler, y_scaler), (input_features, output_features) = prepare_data(config)
    
    # Create model
    model = TemporalConvNet(
        input_size=len(input_features),
        output_size=len(output_features),
        num_channels=config['num_channels'],
        kernel_size=config['kernel_size'],
        dropout=config['dropout']
    )
    
    # Train model
    best_model_state, best_val_loss = train_model(model, train_loader, val_loader, config)
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluate model
    metrics, zscored_metrics = evaluate_model(model, test_loader, y_scaler, output_features, plots_dir)
    
    # Print results
    print("\nFinal Results:")
    for angle, angle_metrics in metrics.items():
        print(f"\n{angle}:")
        for metric, value in angle_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Save model and results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_save_path = models_dir / f'tcn_nowindow_{timestamp}.pth'
    
    torch.save({
        'model_state_dict': best_model_state,
        'config': config,
        'input_features': input_features,
        'output_features': output_features,
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'metrics': metrics,
        'zscored_metrics': zscored_metrics,
        'test_trial_indices': test_loader.dataset.tensors[0][:, 0].numpy()  # Save test trial indices
    }, model_save_path)
    
    print(f"\nModel and results saved to: {model_save_path}")

if __name__ == "__main__":
    main() 