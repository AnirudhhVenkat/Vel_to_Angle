import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
import torch
from torch.utils.data import (
    DataLoader, 
    TensorDataset, 
    Dataset,
    random_split
)
import os
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def filter_frames(df):
    """Filter frames to only include frames 400-1000 from each trial."""
    # Get trial indices (assuming 1400 frames per trial)
    trial_size = 1400
    num_trials = len(df) // trial_size
    
    # Create a mask for frames 400-1000 in each trial
    mask = np.zeros(len(df), dtype=bool)
    for trial in range(num_trials):
        start_idx = trial * trial_size + 400
        end_idx = trial * trial_size + 1000
        mask[start_idx:end_idx] = True
    
    # Apply the mask
    filtered_df = df[mask].copy()
    print(f"Filtered data from {len(df)} to {len(filtered_df)} frames")
    return filtered_df

def create_past_windows_per_trial(trials, targets, window_size):
    """Create windows within each trial using a stride of 1.
    
    Args:
        trials: List of trial features, each of shape (trial_length, num_features)
        targets: List of trial targets, each of shape (trial_length, num_targets)
        window_size: Size of each window
    
    Returns:
        X: Windowed features of shape (num_windows, window_size, num_features)
        y: Windowed targets of shape (num_windows, window_size, num_targets)
    """
    # Calculate total windows with stride of 1
    total_windows = sum(max(0, len(trial) - window_size + 1) for trial in trials)
    X = np.zeros((total_windows, window_size, trials[0].shape[1]), dtype=np.float32)
    y = np.zeros((total_windows, window_size, targets[0].shape[1]), dtype=np.float32)
    
    current_idx = 0
    for trial, target in zip(trials, targets):
        if len(trial) < window_size:
            continue
            
        # Use stride of 1 for overlapping windows
        num_windows = len(trial) - window_size + 1
        for i in range(num_windows):
            start_idx = i  # Stride of 1
            end_idx = start_idx + window_size
            
            if end_idx > len(trial):
                break
                
            X[current_idx] = trial[start_idx:end_idx]
            y[current_idx] = target[start_idx:end_idx]
            current_idx += 1
    
    return X[:current_idx], y[:current_idx]

def calculate_angular_velocities(velocities):
    """Calculate angular velocities from velocity components.
    
    Args:
        velocities: numpy array of shape (n_frames, 3) containing [x_vel, y_vel, z_vel]
    
    Returns:
        Dictionary containing angular velocities
    
    Raises:
        ValueError: If input data is invalid or calculation fails
    """
    try:
        # Input validation
        if not isinstance(velocities, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if velocities.shape[1] != 3:
            raise ValueError(f"Expected 3 velocity components, got {velocities.shape[1]}")
        if not np.isfinite(velocities).all():
            raise ValueError("Input contains non-finite values")
            
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
        xy_angular_vel = pd.Series(xy_angular_vel).rolling(window=window_size, min_periods=window_size, center=True).mean().values
        xz_angular_vel = pd.Series(xz_angular_vel).rolling(window=window_size, min_periods=window_size, center=True).mean().values
        yz_angular_vel = pd.Series(yz_angular_vel).rolling(window=window_size, min_periods=window_size, center=True).mean().values
        
        # Handle NaN values from rolling mean
        xy_angular_vel = pd.Series(xy_angular_vel).fillna(method='ffill').fillna(method='bfill').values
        xz_angular_vel = pd.Series(xz_angular_vel).fillna(method='ffill').fillna(method='bfill').values
        yz_angular_vel = pd.Series(yz_angular_vel).fillna(method='ffill').fillna(method='bfill').values
        
        # Verify no NaN values remain
        if not np.isfinite([xy_angular_vel, xz_angular_vel, yz_angular_vel]).all():
            raise ValueError("Non-finite values in calculated angular velocities")
        
        # Ensure all outputs are float32
        return {
            'xy_angular_vel': xy_angular_vel.astype(np.float32),
            'xz_angular_vel': xz_angular_vel.astype(np.float32),
            'yz_angular_vel': yz_angular_vel.astype(np.float32)
        }
    except Exception as e:
        raise ValueError(f"Error calculating angular velocities: {str(e)}")

def calculate_enhanced_features(data, config):
    """Calculate enhanced features from raw velocities, focusing on most informative features.
    
    Args:
        data: DataFrame containing raw data
        config: Configuration dictionary with optional parameters:
            - window_sizes: List of window sizes for moving averages [default: [5, 10, 20]]
            - min_periods: Minimum periods for moving averages [default: None (use window size)]
    
    Returns:
        DataFrame with enhanced features
    """
    # Get configuration parameters
    window_sizes = config.get('window_sizes', [5, 10, 20])
    min_periods = config.get('min_periods', None)  # If None, will use window size
    
    # First reshape data into trials to handle boundaries correctly
    trial_length = 600  # After filtering (frames 400-1000, so 600 frames per trial)
    num_trials = len(data) // trial_length
    
    # Verify data length is multiple of trial_length
    if len(data) % trial_length != 0:
        raise ValueError(f"Data length ({len(data)}) is not a multiple of trial_length ({trial_length})")
    
    features_list = []
    
    # Process each trial separately to avoid boundary effects
    for trial in range(num_trials):
        start_idx = trial * trial_length
        end_idx = (trial + 1) * trial_length
        trial_data = data.iloc[start_idx:end_idx].copy()
        
        # Check for NaN values in raw data
        if trial_data[['x_vel', 'y_vel', 'z_vel']].isna().any().any():
            raise ValueError(f"NaN values found in raw velocities for trial {trial}")
        
        trial_features = pd.DataFrame()
        
        # Original velocities
        trial_features['x_vel'] = trial_data['x_vel']
        trial_features['z_vel'] = trial_data['z_vel']
        
        # Moving averages for x and z velocities
        for window in window_sizes:
            # Use window size as min_periods if not specified
            periods = min_periods if min_periods is not None else window
            
            # Calculate moving averages within trial boundaries
            trial_features[f'x_vel_ma{window}'] = (
                trial_data['x_vel']
                .rolling(window=window, center=True, min_periods=periods)
                .mean()
            )
            trial_features[f'z_vel_ma{window}'] = (
                trial_data['z_vel']
                .rolling(window=window, center=True, min_periods=periods)
                .mean()
            )
        
        # Combined velocity features
        trial_features['velocity_magnitude'] = np.sqrt(
            trial_data['x_vel']**2 + 
            trial_data['y_vel']**2 + 
            trial_data['z_vel']**2
        )
        trial_features['xz_velocity'] = np.sqrt(
            trial_data['x_vel']**2 + 
            trial_data['z_vel']**2
        )
        
        # Handle any NaN values at edges
        trial_features = trial_features.fillna(method='ffill').fillna(method='bfill')
        
        features_list.append(trial_features)
    
    # Concatenate all trials
    features = pd.concat(features_list, axis=0, ignore_index=True)
    
    # Verify no NaN values were introduced
    if features.isna().any().any():
        raise ValueError("NaN values found in calculated features")
    
    # List of features in the order we want
    all_features = [
        # Z-velocity features (strongest correlations)
        'z_vel', 'z_vel_ma5', 'z_vel_ma10', 'z_vel_ma20',
        # X-velocity features (moderate correlations)
        'x_vel', 'x_vel_ma5', 'x_vel_ma10', 'x_vel_ma20',
        # Combined velocities
        'velocity_magnitude', 'xz_velocity'
    ]
    
    # Verify all expected features are present
    missing_features = set(all_features) - set(features.columns)
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    return features[all_features]

def prepare_data(config):
    """Prepare data for training, validation, and testing.
    
    Args:
        config: Configuration dictionary containing:
            - data_path: Path to data file
            - use_windows: Whether to use windowed data (default: False)
            - window_size: Size of windows if use_windows is True (default: 50)
            - Other standard config parameters...
    
    Returns:
        train_loader, val_loader, test_loader: DataLoaders for each split
        feature_scaler, target_scaler: Fitted scalers for features and targets
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Get configuration parameters
    norm_strategy = config.get('normalization', 'standard')
    use_windows = config.get('use_windows', False)  # Default to False - only use windows when explicitly requested
    window_size = config.get('window_size', 50) if use_windows else None
    
    if norm_strategy not in ['standard', 'minmax']:
        raise ValueError(f"Unknown normalization strategy: {norm_strategy}")
    
    # Load data
    data = pd.read_csv(config['data_path'])
    
    # Filter for P9RT genotype
    print(f"Total samples before filtering: {len(data)}")
    data = data[data['genotype'] == 'P9RT'].copy()
    print(f"Total samples after filtering for P9RT: {len(data)}")
    
    if len(data) == 0:
        raise ValueError("No data found for genotype P9RT")
    
    # Filter frames to only include frames 400-1000 from each trial
    data = filter_frames(data)
    
    # Verify data exists and has expected columns
    if data.empty:
        raise ValueError("Empty dataset")
    
    required_columns = ['x_vel', 'y_vel', 'z_vel'] + config['joint_angle_columns']
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Calculate enhanced features
    features = calculate_enhanced_features(data, config)
    
    # Select only the specified velocity features
    selected_features = config['velocity_features']
    if not all(feat in features.columns for feat in selected_features):
        raise ValueError("Not all specified velocity features are available")
    features = features[selected_features]
    
    # Select only the specified joint angles
    joint_angles = data[config['joint_angle_columns']]
    
    # Use correct trial length after filtering (600 frames per trial)
    trial_length = 600  # After filtering frames 400-1000
    num_trials = len(features) // trial_length
    
    # Verify data length
    if len(features) % trial_length != 0:
        raise ValueError(f"Data length ({len(features)}) is not a multiple of trial_length ({trial_length})")
    
    print(f"\nTotal number of trials: {num_trials}")
    
    # Create trial indices and shuffle
    trial_indices = np.arange(num_trials)
    np.random.seed(42)  # For reproducible splits
    np.random.shuffle(trial_indices)
    
    # Fixed test size of 12 trials
    test_trials = 12
    
    # Ensure we have enough trials
    if num_trials < 24:  # Need at least 24 trials (12 test + minimum for train/val)
        raise ValueError(f"Not enough trials ({num_trials}). Need at least 24 trials for proper splitting.")
    
    # Calculate remaining trials for train/val
    remaining_trials = num_trials - test_trials
    train_trials = int(0.8 * remaining_trials)  # 80% of remaining for training
    val_trials = remaining_trials - train_trials  # Rest for validation
    
    # Split indices
    train_indices = trial_indices[:train_trials]
    val_indices = trial_indices[train_trials:train_trials + val_trials]
    test_indices = trial_indices[-test_trials:]  # Take last 12 trials for test
    
    print(f"\nNumber of trials in each split:")
    print(f"Train: {len(train_indices)} trials")
    print(f"Val: {len(val_indices)} trials")
    print(f"Test: {len(test_indices)} trials (fixed at 12)")
    
    # Reshape to trials
    features_array = features.values.reshape(num_trials, trial_length, -1)
    targets_array = joint_angles.values.reshape(num_trials, trial_length, -1)
    
    print(f"\nTrial shape: {features_array.shape}")
    print(f"Number of input features: {features_array.shape[-1]}")
    print(f"Number of target joints: {targets_array.shape[-1]}")
    
    # Standardize features and targets using only training data
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    # Fit scalers on training data only
    train_features_2d = features_array[train_indices].reshape(-1, features_array.shape[-1])
    train_targets_2d = targets_array[train_indices].reshape(-1, targets_array.shape[-1])
    
    # Check for NaN or infinite values before scaling
    if np.any(~np.isfinite(train_features_2d)):
        raise ValueError("Non-finite values found in training features")
    if np.any(~np.isfinite(train_targets_2d)):
        raise ValueError("Non-finite values found in training targets")
    
    feature_scaler.fit(train_features_2d)
    target_scaler.fit(train_targets_2d)
    
    # Transform all data
    features_array_scaled = np.zeros_like(features_array)
    targets_array_scaled = np.zeros_like(targets_array)
    
    # Apply scaling to each split separately
    for trial_idx in range(num_trials):
        features_array_scaled[trial_idx] = feature_scaler.transform(features_array[trial_idx])
        targets_array_scaled[trial_idx] = target_scaler.transform(targets_array[trial_idx])
    
    # Check for NaN or infinite values after scaling
    if np.any(~np.isfinite(features_array_scaled)):
        raise ValueError("Non-finite values found after scaling features")
    if np.any(~np.isfinite(targets_array_scaled)):
        raise ValueError("Non-finite values found after scaling targets")
    
    # Create datasets for each split
    train_features = features_array_scaled[train_indices]
    train_targets = targets_array_scaled[train_indices]
    val_features = features_array_scaled[val_indices]
    val_targets = targets_array_scaled[val_indices]
    test_features = features_array_scaled[test_indices]
    test_targets = targets_array_scaled[test_indices]
    
    if use_windows:
        print(f"\nCreating windowed datasets with window_size={window_size}")
        # Create windows for each split
        train_features, train_targets = create_past_windows_per_trial(
            train_features, train_targets, window_size
        )
        val_features, val_targets = create_past_windows_per_trial(
            val_features, val_targets, window_size
        )
        test_features, test_targets = create_past_windows_per_trial(
            test_features, test_targets, window_size
        )
        print("\nWindow dataset sizes (windows x window_size x features):")
    else:
        print("\nUsing full sequences (no windowing)")
        print("\nDataset sizes (trials x frames x features):")
    
    # Convert to tensors
    train_features = torch.FloatTensor(train_features)
    train_targets = torch.FloatTensor(train_targets)
    val_features = torch.FloatTensor(val_features)
    val_targets = torch.FloatTensor(val_targets)
    test_features = torch.FloatTensor(test_features)
    test_targets = torch.FloatTensor(test_targets)
    
    print(f"Train: {train_features.shape}")
    print(f"Val: {val_features.shape}")
    print(f"Test: {test_features.shape}")
    
    # Create datasets
    train_dataset = TensorDataset(train_features, train_targets)
    val_dataset = TensorDataset(val_features, val_targets)
    test_dataset = TensorDataset(test_features, test_targets)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, feature_scaler, target_scaler

def create_windows(features, targets, window_size):
    """Create non-overlapping windows from features and targets."""
    num_trials, trial_length, num_features = features.shape
    num_windows = trial_length // window_size
    
    # Reshape to create windows
    windowed_features = features[:, :num_windows * window_size].reshape(
        num_trials * num_windows, window_size, num_features
    )
    windowed_targets = targets[:, :num_windows * window_size].reshape(
        num_trials * num_windows, window_size, -1
    )
    
    return windowed_features, windowed_targets
