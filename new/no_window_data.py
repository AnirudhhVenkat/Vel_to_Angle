import torch
import torch.nn as nn
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
import traceback
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import signal
import random
import os

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

class WholeTrialDataset(Dataset):
    """
    Dataset that works with entire trials instead of windowed sequences.
    This is more suitable for TimeMixer and TimeXer models which handle sequential
    data internally without requiring explicit windowing.
    """
    def __init__(self, X, y, trial_indices=None, filtered_to_original=None):
        """
        Initialize the dataset with entire trials.
        
        Args:
            X: Input features of shape [num_trials, trial_length, num_features]
            y: Target values of shape [num_trials, trial_length, num_outputs]
            trial_indices: List of trial indices (for traceability)
            filtered_to_original: Mapping from filtered trial indices to original indices
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.trial_indices = trial_indices
        self.filtered_to_original = filtered_to_original
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Return the entire trial and its target, along with the trial index
        X_trial = self.X[idx]
        y_trial = self.y[idx]
        
        # Get original trial index if available
        trial_idx = self.trial_indices[idx] if self.trial_indices is not None else idx
        original_idx = self.filtered_to_original.get(trial_idx, trial_idx) if self.filtered_to_original else trial_idx
        
        return X_trial, y_trial, torch.tensor(original_idx, dtype=torch.long)

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
        
        # Handle 3D input (trials, time, features)
        if X.ndim == 3:
            # Reshape to 2D to fit
            orig_shape = X.shape
            X = X.reshape(-1, X.shape[-1])
            self.means = np.mean(X, axis=0)
            self.stds = np.std(X, axis=0)
            # Handle constant features
            self.stds[self.stds == 0] = 1
        else:
            # Standard 2D case
            self.means = np.mean(X, axis=0)
            self.stds = np.std(X, axis=0)
            # Handle constant features
            self.stds[self.stds == 0] = 1
        return self
    
    def transform(self, X):
        """Transform the data."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Handle 3D input (trials, time, features)
        if X.ndim == 3:
            # Apply normalization along the feature dimension
            return (X - self.means) / self.stds
        else:
            # Standard 2D case
            return (X - self.means) / self.stds
    
    def fit_transform(self, X):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """Convert back to original scale."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return X * self.stds + self.means

def filter_trial_frames(X_trials, y_trials, start_frame=350, end_frame=1000):
    """
    Filter frames within each trial to focus on a specific segment.
    Unlike the windowed approach, this preserves whole trials but trims them.
    
    Args:
        X_trials: Input features with shape [num_trials, trial_length, num_features]
        y_trials: Target values with shape [num_trials, trial_length, num_outputs]
        start_frame: Start frame to keep (inclusive)
        end_frame: End frame to keep (inclusive)
        
    Returns:
        Tuple of filtered X_trials, y_trials
    """
    filtered_X = []
    filtered_y = []
    
    for i in range(len(X_trials)):
        # Extract the frames we want to keep
        filtered_X.append(X_trials[i, start_frame:end_frame+1, :])
        filtered_y.append(y_trials[i, start_frame:end_frame+1, :])
    
    # Stack trials back into 3D arrays
    filtered_X = np.stack(filtered_X)
    filtered_y = np.stack(filtered_y)
    
    print(f"Filtered data shapes:")
    print(f"X: {filtered_X.shape}")
    print(f"y: {filtered_y.shape}")
    
    return filtered_X, filtered_y

def calculate_enhanced_features(trials_data, frames_per_trial):
    """
    Calculate enhanced features including lagged velocities and moving averages.
    This preserves the trial-based structure of the data.
    """
    enhanced_trials = []
    
    # Process each trial
    for trial_data in tqdm(trials_data, desc="Calculating enhanced features"):
        # Extract velocity data
        trial_features = pd.DataFrame()
        
        # First, copy all angle columns from the original data
        # This ensures we preserve the output features
        for col in trial_data.columns:
            # Copy all angle columns and other non-velocity columns we want to keep
            if col.endswith('_flex') or col.endswith('_rot') or col.endswith('_abduct') or col == 'genotype':
                trial_features[col] = trial_data[col].values
        
        # Copy velocity columns
        if 'x_vel' in trial_data.columns:
            trial_features['x_vel'] = trial_data['x_vel'].values
            trial_features['y_vel'] = trial_data['y_vel'].values
            trial_features['z_vel'] = trial_data['z_vel'].values
            
            # Calculate velocity magnitude
            trial_features['vel_mag'] = np.sqrt(
                trial_features['x_vel']**2 + 
                trial_features['y_vel']**2 + 
                trial_features['z_vel']**2
            )
            
            # Calculate acceleration
            dt = 1/30  # 30 Hz sampling rate
            trial_features['x_acc'] = np.gradient(trial_features['x_vel'].values, dt)
            trial_features['y_acc'] = np.gradient(trial_features['y_vel'].values, dt)
            trial_features['z_acc'] = np.gradient(trial_features['z_vel'].values, dt)
            
            # Calculate acceleration magnitude
            trial_features['acc_mag'] = np.sqrt(
                trial_features['x_acc']**2 + 
                trial_features['y_acc']**2 + 
                trial_features['z_acc']**2
            )
            
            # Calculate moving averages for velocity
            for window in [5, 10, 20]:
                for coord in ['x', 'y', 'z']:
                    # Calculate moving average
                    ma = trial_features[f'{coord}_vel'].rolling(window=window, center=True, min_periods=1).mean()
                    trial_features[f'{coord}_vel_ma{window}'] = ma
                
                # Calculate moving average for velocity magnitude
                ma = trial_features['vel_mag'].rolling(window=window, center=True, min_periods=1).mean()
                trial_features[f'vel_mag_ma{window}'] = ma
            
            # Calculate moving averages for acceleration
            for window in [5, 10, 20]:
                for coord in ['x', 'y', 'z']:
                    # Calculate moving average
                    ma = trial_features[f'{coord}_acc'].rolling(window=window, center=True, min_periods=1).mean()
                    trial_features[f'{coord}_acc_ma{window}'] = ma
                
                # Calculate moving average for acceleration magnitude
                ma = trial_features['acc_mag'].rolling(window=window, center=True, min_periods=1).mean()
                trial_features[f'acc_mag_ma{window}'] = ma
            
            # Calculate lagged velocities
            for lag in [1, 2, 3, 4, 5]:
                for coord in ['x', 'y', 'z']:
                    # Get velocity array
                    vel = trial_features[f'{coord}_vel'].values
                    
                    # Calculate lagged velocity
                    lagged_vel = np.zeros_like(vel)
                    if len(vel) > lag:  # Make sure array is not empty and has enough elements
                        lagged_vel[lag:] = vel[:-lag]
                        trial_features[f'{coord}_vel_lag{lag}'] = lagged_vel
                    else:
                        # If array is too small, just use zeros
                        trial_features[f'{coord}_vel_lag{lag}'] = 0
        
        # Add PSD features
        fs = 200  # 200Hz sampling frequency
        for vel in ['x_vel', 'y_vel', 'z_vel']:
            if vel in trial_features.columns:
                # Calculate PSD using Welch's method
                frequencies, psd = signal.welch(trial_features[vel].values, fs=fs, nperseg=fs)
                
                # Total power
                total_power = np.sum(psd)
                trial_features[f'{vel}_psd'] = total_power
        
        # Add trial to enhanced trials
        enhanced_trials.append(trial_features)
    
    return enhanced_trials

def get_available_data_path(genotype):
    """Try multiple possible data paths and return the first available one based on genotype."""
    if genotype == 'ES':
        possible_paths = [
            r"Z:\Divya\TEMP_transfers\toAni\4_StopProjectData_forES\df_preproc_fly_centric.parquet"
        ]
    else:
        possible_paths = [
            "Z:/Divya/TEMP_transfers/toAni/BPN_P9LT_P9RT_flyCoords.csv",  # Network drive path
            "/Users/anivenkat/Downloads/BPN_P9LT_P9RT_flyCoords.csv",      # Local Mac path
           # "C:/Users/bidayelab/Downloads/BPN_P9LT_P9RT_flyCoords.csv"     # Local Windows path
        ]
    
    for path in possible_paths:
        if Path(path).exists():
            print(f"Using data file: {path}")
            return path
    
    raise FileNotFoundError(f"Could not find the data file for genotype {genotype}")

def prepare_data_no_windows(data_path, input_features, output_features, pred_len, output_dir=None, leg_name=None, fixed_trial_splits=None, fixed_filtered_to_original=None):
    """
    Prepare data for TimeMixer and TimeXer models without windowing.
    This function processes the data to maintain whole trials rather than sliding windows.
    
    Args:
        data_path: Path to the raw data file
        input_features: List of input feature names
        output_features: List of output feature names
        pred_len: Prediction length for models
        output_dir: Directory to save outputs
        leg_name: Name of the leg being processed
        fixed_trial_splits: Pre-defined trial splits (optional)
        fixed_filtered_to_original: Mapping from filtered to original trial indices (optional)
        
    Returns:
        Tuple containing processed data and loaders for whole-trial prediction
    """
    print(f"\nPreparing data from {data_path} for non-windowed processing...")
    
    # Check if we're using fixed trial splits
    using_fixed_splits = fixed_trial_splits is not None and fixed_filtered_to_original is not None
    if using_fixed_splits:
        print("\nUsing provided fixed trial splits and filtered-to-original mapping")
        print(f"  Train: {len(fixed_trial_splits['train'])} trials")
        print(f"  Validation: {len(fixed_trial_splits['val'])} trials") 
        print(f"  Test: {len(fixed_trial_splits['test'])} trials")
        
        # Convert string keys back to integers if they came from JSON
        if isinstance(next(iter(fixed_filtered_to_original.keys()), '0'), str):
            fixed_filtered_to_original = {int(k): v for k, v in fixed_filtered_to_original.items()}
            print("  Converted filtered-to-original mapping keys from strings to integers")
    else:
        print("\nGenerating new trial splits")
    
    # Load main data
    if str(data_path).endswith('.csv'):
        df = pd.read_csv(data_path)
    elif str(data_path).endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    print(f"Main data loaded with shape: {df.shape}")
    
    # Print genotype distribution
    print("\nGenotype distribution in data:")
    genotype_counts = df['genotype'].value_counts()
    for genotype, count in genotype_counts.items():
        print(f"  {genotype}: {count} frames")
    
    # Split data into trials (1400 frames per trial)
    original_trial_size = 1400
    num_trials = len(df) // original_trial_size
    
    print(f"Original number of trials: {num_trials}")
    
    # Create list of trial DataFrames
    all_trials = []
    for i in range(num_trials):
        trial_start = i * original_trial_size
        trial_end = (i + 1) * original_trial_size
        
        # Make sure we have enough frames for a complete trial
        if trial_end <= len(df):
            trial_data = df.iloc[trial_start:trial_end].copy()
            all_trials.append(trial_data)
    
    print(f"Created {len(all_trials)} complete trials")
    
    # Apply velocity thresholds to filter trials
    filtered_trials = []
    velocity_filtered_to_original = {}  # Rename to avoid confusion
    filtered_trial_idx = 0
    
    # Track passing and failing trials for debugging
    passing_trials = []
    failing_trials = []
    
    print("\nApplying velocity thresholds:")
    for i, trial_data in enumerate(all_trials):
        # Get genotype for this trial
        genotype = trial_data['genotype'].iloc[0]
        
        # Apply different velocity thresholds based on genotype
        passes_threshold = False
        avg_vel = 0
        vel_type = ""
        
        if genotype == 'BPN':
            # BPN: Check if average x-velocity exceeds 5 mm/s
            frames = trial_data.iloc[350:1000]
            avg_vel = frames['x_vel'].abs().mean()
            vel_type = "x"
            passes_threshold = avg_vel >= 5
        elif genotype in ['P9RT', 'P9LT']:
            # P9RT/P9LT: Check if average z-velocity exceeds 3 mm/s
            frames = trial_data.iloc[350:1000]
            avg_vel = frames['z_vel'].abs().mean()
            vel_type = "z"
            passes_threshold = avg_vel >= 3
        elif genotype == 'ES':
            # ES: Check if average z-velocity exceeds 5 mm/s
            frames = trial_data.iloc[0:650]
            avg_vel = frames['x_vel'].abs().mean()
            vel_type = "x"
            passes_threshold = avg_vel >= 5
        
        # Debug print for each trial
        print(f"Trial {i}: genotype={genotype}, avg {vel_type}-vel={avg_vel:.2f}, passes_threshold={passes_threshold}")
        
        # Track passing/failing trials
        if passes_threshold:
            passing_trials.append((i, genotype, avg_vel, vel_type))
            
            # Only keep trials that pass the threshold
            filtered_trials.append(trial_data)
            velocity_filtered_to_original[filtered_trial_idx] = i
            filtered_trial_idx += 1
        else:
            failing_trials.append((i, genotype, avg_vel, vel_type))
    
    # Calculate enhanced features for filtered trials
    print("\nCalculating enhanced features for filtered trials...")
    enhanced_trials = calculate_enhanced_features(filtered_trials, frames_per_trial=original_trial_size)
    
    # Extract input and output features
    X_trials = []
    y_trials = []
    
    for trial_df in enhanced_trials:
        X_trial = trial_df[input_features].values
        y_trial = trial_df[output_features].values
        
        # Add to lists
        X_trials.append(X_trial)
        y_trials.append(y_trial)
    
    # Convert to 3D numpy arrays [num_trials, trial_length, num_features]
    X_trials = np.array(X_trials)
    y_trials = np.array(y_trials)
    
    print(f"\nTrial data shapes:")
    print(f"X_trials: {X_trials.shape}")
    print(f"y_trials: {y_trials.shape}")
    
    # Filter frames to focus on relevant segment
    X_filtered, y_filtered = filter_trial_frames(X_trials, y_trials, start_frame=350, end_frame=1000)
    
    # If fixed trial splits are provided, use those instead of creating new ones
    if using_fixed_splits:
        trial_splits = fixed_trial_splits
        filtered_to_original = fixed_filtered_to_original
    else:
        # Create stratified splits
        trial_splits = {'train': [], 'val': [], 'test': []}
        
        # Group trials by genotype for stratified sampling
        genotype_trials = {}
        for i, trial_data in enumerate(filtered_trials):
            genotype = trial_data['genotype'].iloc[0]
            if genotype not in genotype_trials:
                genotype_trials[genotype] = []
            genotype_trials[genotype].append((i, trial_data))
        
        for genotype, genotype_trial_data in genotype_trials.items():
            # Shuffle trials for this genotype
            indices = list(range(len(genotype_trial_data)))
            np.random.shuffle(indices)
            
            # Calculate split sizes (70% train, 15% val, 15% test)
            n_trials = len(genotype_trial_data)
            n_train = int(0.7 * n_trials)
            n_val = int(0.15 * n_trials)
            
            # Split trials
            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train+n_val]
            test_indices = indices[n_train+n_val:]
            
            # Add to respective lists
            for idx in train_indices:
                filtered_idx, _ = genotype_trial_data[idx]
                trial_splits['train'].append(filtered_idx)
            
            for idx in val_indices:
                filtered_idx, _ = genotype_trial_data[idx]
                trial_splits['val'].append(filtered_idx)
            
            for idx in test_indices:
                filtered_idx, _ = genotype_trial_data[idx]
                trial_splits['test'].append(filtered_idx)
        
        filtered_to_original = velocity_filtered_to_original
    
    # Split data based on trial indices
    X_train = np.array([X_filtered[i] for i in trial_splits['train']])
    y_train = np.array([y_filtered[i] for i in trial_splits['train']])
    
    X_val = np.array([X_filtered[i] for i in trial_splits['val']])
    y_val = np.array([y_filtered[i] for i in trial_splits['val']])
    
    X_test = np.array([X_filtered[i] for i in trial_splits['test']])
    y_test = np.array([y_filtered[i] for i in trial_splits['test']])
    
    print("\nData shapes after splitting:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    
    # Scale features
    X_scaler = ZScoreScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_val = X_scaler.transform(X_val)
    X_test = X_scaler.transform(X_test)
    
    # Scale targets
    y_scaler = ZScoreScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_val = y_scaler.transform(y_val)
    y_test = y_scaler.transform(y_test)
    
    # Create DataLoaders using WholeTrialDataset
    train_dataset = WholeTrialDataset(
        X_train, y_train, trial_indices=trial_splits['train'], filtered_to_original=filtered_to_original
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,  # Use batch_size=1 for whole trials
        shuffle=True,  # Shuffle trials during training
        num_workers=0
    )
    
    val_dataset = WholeTrialDataset(
        X_val, y_val, trial_indices=trial_splits['val'], filtered_to_original=filtered_to_original
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1,  # Use batch_size=1 for whole trials
        shuffle=False,  # No need to shuffle for validation
        num_workers=0
    )
    
    test_dataset = WholeTrialDataset(
        X_test, y_test, trial_indices=trial_splits['test'], filtered_to_original=filtered_to_original
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1,  # Use batch_size=1 for whole trials
        shuffle=False,  # No need to shuffle for testing
        num_workers=0
    )
    
    print("\nDataLoaders created:")
    print(f"  Train: {len(train_loader.dataset)} trials")
    print(f"  Validation: {len(val_loader.dataset)} trials")
    print(f"  Test: {len(test_loader.dataset)} trials")
    print(f"  Using batch_size=1 for whole-trial processing (no windowing)")
    
    return (
        df, filtered_trials, X_train, y_train, X_val, y_val, X_test, y_test,
        train_loader, val_loader, test_loader,
        X_scaler, y_scaler, trial_splits, filtered_to_original
    ) 