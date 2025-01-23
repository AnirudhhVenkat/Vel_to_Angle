import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import os

class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def filter_frames(df):
    """Filter frames efficiently using numpy operations"""
    f_0, f_f, f_trl = 400, 1000, 1400
    mask = ((df['fnum'].values % f_trl) >= f_0) & ((df['fnum'].values % f_trl) < f_f)
    return df[mask]

def create_past_windows_per_trial(trials, targets, window_size, stride=1):
    """Create windows within each trial using numpy operations"""
    X, y = [], []
    
    # Pre-allocate lists with estimated size
    total_windows = sum((len(trial) - window_size) // stride + 1 for trial in trials)
    X = np.zeros((total_windows, window_size, trials[0].shape[1]), dtype=np.float32)
    y = np.zeros((total_windows, targets[0].shape[1]), dtype=np.float32)
    
    current_idx = 0
    for trial, target in zip(trials, targets):
        if len(trial) <= window_size:
            continue
            
        num_windows = (len(trial) - window_size) // stride + 1
        for i in range(num_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size
            
            if end_idx >= len(trial):
                break
                
            X[current_idx] = trial[start_idx:end_idx]
            y[current_idx] = target[end_idx-1]
            current_idx += 1
    
    return X[:current_idx], y[:current_idx]

def calculate_angular_velocities(velocities):
    """
    Calculate angular velocities from velocity components.
    
    Args:
        velocities: numpy array of shape (n_frames, 3) containing [x_vel, y_vel, z_vel]
    
    Returns:
        Dictionary containing angular velocities
    """
    try:
        # Ensure input is float32
        velocities = velocities.astype(np.float32)
        
        # Calculate angles at each timestep
        xy_angles = np.arctan2(velocities[:, 1], velocities[:, 0])  # y_vel/x_vel
        xz_angles = np.arctan2(velocities[:, 2], velocities[:, 0])  # z_vel/x_vel
        yz_angles = np.arctan2(velocities[:, 2], velocities[:, 1])  # z_vel/y_vel
        
        # Calculate angular velocities (rate of change of angles)
        xy_angular_vel = np.diff(xy_angles, prepend=xy_angles[0])
        xz_angular_vel = np.diff(xz_angles, prepend=xz_angles[0])
        yz_angular_vel = np.diff(yz_angles, prepend=yz_angles[0])
        
        # Handle discontinuities in angular velocities
        xy_angular_vel = np.where(abs(xy_angular_vel) > np.pi, 
                                xy_angular_vel - np.sign(xy_angular_vel) * 2 * np.pi, 
                                xy_angular_vel)
        xz_angular_vel = np.where(abs(xz_angular_vel) > np.pi,
                                xz_angular_vel - np.sign(xz_angular_vel) * 2 * np.pi,
                                xz_angular_vel)
        yz_angular_vel = np.where(abs(yz_angular_vel) > np.pi,
                                yz_angular_vel - np.sign(yz_angular_vel) * 2 * np.pi,
                                yz_angular_vel)
        
        # Apply smoothing to reduce noise
        window_size = 5
        xy_angular_vel = pd.Series(xy_angular_vel).rolling(window=window_size, min_periods=1, center=True).mean().values
        xz_angular_vel = pd.Series(xz_angular_vel).rolling(window=window_size, min_periods=1, center=True).mean().values
        yz_angular_vel = pd.Series(yz_angular_vel).rolling(window=window_size, min_periods=1, center=True).mean().values
        
        # Ensure all outputs are float32
        return {
            'xy_angular_vel': xy_angular_vel.astype(np.float32),
            'xz_angular_vel': xz_angular_vel.astype(np.float32),
            'yz_angular_vel': yz_angular_vel.astype(np.float32)
        }
    except Exception as e:
        print(f"Error calculating angular velocities: {e}")
        # Return zero velocities as fallback
        zeros = np.zeros(len(velocities), dtype=np.float32)
        return {
            'xy_angular_vel': zeros,
            'xz_angular_vel': zeros,
            'yz_angular_vel': zeros
        }

def prepare_data(config):
    """
    Prepare data for model training with enhanced features.
    
    Args:
        config: Dictionary containing configuration parameters
        
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects for training, validation and testing
    """
    try:
        # Load data
        velocity_cols = ['x_vel', 'y_vel', 'z_vel']
        joint_cols = ['L2B_rot', 'R3A_rot', 'R3A_flex', 'R1B_rot', 'R2B_rot', 'L3A_rot']
        
        df = pd.read_csv(
            config['data_path'],
            usecols=velocity_cols + joint_cols,
            dtype={col: np.float32 for col in velocity_cols + joint_cols},
            engine='python'
        )
        
        # Calculate enhanced features
        velocities = df[velocity_cols].values.astype(np.float32)
        features_dict = {}
        
        # 1. Moving averages (9 features)
        window_sizes = [5, 10, 20]
        for w in window_sizes:
            for i, vel_col in enumerate(velocity_cols):
                ma = pd.Series(velocities[:, i]).rolling(window=w, min_periods=1).mean().values
                features_dict[f'{vel_col}_ma{w}'] = ma.astype(np.float32)
        
        # 2. Velocity magnitude (1 feature)
        velocity_magnitude = np.sqrt(np.sum(velocities**2, axis=1))
        features_dict['velocity_magnitude'] = velocity_magnitude.astype(np.float32)
        
        # 3. Planar velocities (2 features)
        xy_velocity = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
        xz_velocity = np.sqrt(velocities[:, 0]**2 + velocities[:, 2]**2)
        features_dict['xy_velocity'] = xy_velocity.astype(np.float32)
        features_dict['xz_velocity'] = xz_velocity.astype(np.float32)
        
        # 4. Original velocities (3 features)
        for i, col in enumerate(velocity_cols):
            features_dict[col] = velocities[:, i]
        
        # 5. Angular velocities (optional - 3 features)
        if config.get('use_angular_velocities', False):
            angular_vel_features = calculate_angular_velocities(velocities)
            features_dict.update(angular_vel_features)
        
        # Convert to DataFrame and standardize
        features_df = pd.DataFrame(features_dict)
        
        # Print feature names and count for verification
        print("\nFeature list:")
        for i, feature in enumerate(features_dict.keys(), 1):
            print(f"{i}. {feature}")
        print(f"\nTotal features: {len(features_dict)}")
        
        # Verify feature count
        expected_features = 18 if config.get('use_angular_velocities', False) else 15
        assert len(features_dict) == expected_features, f"Expected {expected_features} features, but got {len(features_dict)}"
        
        # Standardize features and targets
        features = stats.zscore(features_df.values, axis=0).astype(np.float32)
        targets = stats.zscore(df[joint_cols].values, axis=0).astype(np.float32)
        
        # Split into trials
        trial_size = 600
        num_trials = len(features) // trial_size
        print(f"\nNumber of trials: {num_trials}")
        
        X_trials = [features[i*trial_size:(i+1)*trial_size] for i in range(num_trials)]
        y_trials = [targets[i*trial_size:(i+1)*trial_size] for i in range(num_trials)]
        
        # Create windows within trials
        window_size = config['window_size']
        X, y = create_past_windows_per_trial(X_trials, y_trials, window_size)
        print(f"Created windows with shapes: X={X.shape}, y={y.shape}")
        
        # Split into train, validation, and test sets
        total_samples = len(X)
        test_size = min(config['test_size'], int(total_samples * 0.2))
        train_val_size = total_samples - test_size
        train_size = int(train_val_size * config['split_ratio'])
        val_size = train_val_size - train_size
        
        # Create indices for splitting
        indices = np.random.permutation(total_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
        
        # Split the data
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_val = X[val_indices]
        y_val = y[val_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
        
        print("\nData split sizes:")
        print(f"Train: {len(X_train)} sequences")
        print(f"Validation: {len(X_val)} sequences")
        print(f"Test: {len(X_test)} sequences")
        print(f"\nSequence shapes:")
        print(f"Input shape (batch, seq_len, features): {X_train.shape}")
        print(f"Output shape (batch, num_joints): {y_train.shape}")
        
        # Convert to PyTorch tensors and create datasets
        train_dataset = TimeSeriesDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TimeSeriesDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        test_dataset = TimeSeriesDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0
        )

        # Verify data loader shapes
        for inputs, targets in train_loader:
            print(f"\nBatch shapes:")
            print(f"Input batch shape: {inputs.shape}")
            print(f"Target batch shape: {targets.shape}")
            break

        return train_loader, val_loader, test_loader

    except Exception as e:
        print(f"Error in prepare_data: {e}")
        return None, None, None