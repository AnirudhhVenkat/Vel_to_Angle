import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, Sampler
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
from scipy import signal
import random
from losses import DerivativeLoss
import os
import time

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

# Add a new TrialSampler class to shuffle trials but not sequences within trials
class TrialSampler(Sampler):
    """
    Custom sampler that shuffles trials but keeps sequences within each trial in order.
    This preserves the temporal relationship within trials while allowing randomization
    between different trials during training.
    """
    def __init__(self, dataset, frames_per_trial, sequence_length):
        self.dataset = dataset
        self.frames_per_trial = frames_per_trial
        self.sequence_length = sequence_length
        
        # Calculate the number of sequences per trial
        self.sequences_per_trial = frames_per_trial - sequence_length + 1
        
        # Calculate the number of trials
        self.num_trials = len(dataset) // self.sequences_per_trial
        
        # Verify that the dataset size is a multiple of sequences_per_trial
        if len(dataset) % self.sequences_per_trial != 0:
            print(f"Warning: Dataset size ({len(dataset)}) is not a multiple of sequences per trial ({self.sequences_per_trial})")
            print(f"This may indicate a problem with trial/sequence alignment")
            # Adjust num_trials to handle incomplete trials
            self.num_trials = len(dataset) // self.sequences_per_trial
        
        print(f"TrialSampler initialized with {self.num_trials} trials, {self.sequences_per_trial} sequences per trial")
    
    def __iter__(self):
        # Create a list of trial indices
        trial_indices = list(range(self.num_trials))
        
        # Shuffle the trial indices
        random.shuffle(trial_indices)
        
        # For each trial, yield all its sequence indices in order
        for trial_idx in trial_indices:
            start_idx = trial_idx * self.sequences_per_trial
            for seq_idx in range(self.sequences_per_trial):
                yield start_idx + seq_idx
    
    def __len__(self):
        return len(self.dataset)

class TrialAwareSequenceDataset(Dataset):
    def __init__(self, X, y, sequence_length, frames_per_trial, filtered_to_original=None, trial_indices_list=None):
        # Explicitly cast to float32 to match PyTorch's expected type
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.sequence_length = sequence_length
        self.frames_per_trial = frames_per_trial
        self.filtered_to_original = filtered_to_original
        self.trial_indices_list = trial_indices_list
        
        # Debug: Print information about the mapping
        if filtered_to_original is not None:
            print(f"\nDebug - TrialAwareSequenceDataset initialization:")
            print(f"X shape: {self.X.shape}")
            print(f"frames_per_trial: {frames_per_trial}")
            print(f"Expected number of trials: {len(self.X) // frames_per_trial}")
            print(f"filtered_to_original mapping size: {len(filtered_to_original)}")
            print(f"filtered_to_original keys: {sorted(filtered_to_original.keys())}")
            print(f"filtered_to_original values: {sorted(filtered_to_original.values())}")
        
        # Generate valid indices and their corresponding trial indices
        self.valid_indices, self.trial_indices, self.original_trial_indices = self._get_valid_indices()
        
        # Debug: Print information about the generated indices
        print(f"\nDebug - Generated indices:")
        print(f"Number of valid indices: {len(self.valid_indices)}")
        print(f"Number of trial indices: {len(self.trial_indices)}")
        print(f"Number of original trial indices: {len(self.original_trial_indices)}")
        print(f"Unique trial indices: {len(set(self.trial_indices))}")
        print(f"Unique original trial indices: {len(set(self.original_trial_indices))}")
        
        # Debug: Compare targets in TrialAwareSequenceDataset
        print("\nDebug - Comparing targets in TrialAwareSequenceDataset:")
        for idx in range(min(5, len(self.valid_indices))):
            start_idx = self.valid_indices[idx]
            target_from_valid_idx = self.y[start_idx + self.sequence_length - 1]
            target_from_getitem = self.__getitem__(idx)[1].numpy()  # Call __getitem__ to get the target
            
            # Check if targets are equal
            are_equal = np.allclose(target_from_valid_idx, target_from_getitem)
            print(f"Index {idx}: Target from valid_idx: {target_from_valid_idx}, Target from __getitem__: {target_from_getitem}, Equal: {are_equal}")
            
            if not are_equal:
                print(f"WARNING: Targets don't match for index {idx}!")
                print(f"Difference: {target_from_valid_idx - target_from_getitem}")
        
    def _get_valid_indices(self):
        """Get indices that don't cross trial boundaries and their corresponding trial indices."""
        valid_indices = []
        trial_indices = []  # Store which filtered trial each sequence belongs to
        original_trial_indices = []  # Store which original trial each sequence belongs to
        
        num_trials = len(self.X) // self.frames_per_trial
        
        # Debug: Track which trials are being processed
        trials_with_mapping = 0
        trials_without_mapping = 0
        
        # If trial_indices_list is provided, use it; otherwise, use sequential indices
        if self.trial_indices_list is not None:
            actual_trial_indices = self.trial_indices_list
            print(f"\nDebug - Using provided trial indices list: {sorted(self.trial_indices_list)}")
        else:
            actual_trial_indices = list(range(num_trials))
            print("\nDebug - Using sequential trial indices")
        
        # Verify we have the right number of trials
        if len(actual_trial_indices) != num_trials:
            print(f"\nWARNING: Number of trials in data ({num_trials}) doesn't match length of trial_indices_list ({len(actual_trial_indices)})")
            print(f"This may indicate a mismatch between the data and the trial indices.")
        
        # Process each trial in the dataset
        for seq_idx, actual_trial_idx in enumerate(actual_trial_indices):
            start = seq_idx * self.frames_per_trial
            end = (seq_idx + 1) * self.frames_per_trial - self.sequence_length
            
            # Add all valid starting indices for this trial
            valid_indices.extend(range(start, end + 1))
            
            # Add the actual trial index for each sequence
            trial_indices.extend([actual_trial_idx] * (end - start + 1))
            
            # Add the original trial index if mapping is provided
            if self.filtered_to_original is not None and actual_trial_idx in self.filtered_to_original:
                original_trial_indices.extend([self.filtered_to_original[actual_trial_idx]] * (end - start + 1))
                trials_with_mapping += 1
            else:
                # If no mapping is available, use the actual trial index
                original_trial_indices.extend([actual_trial_idx] * (end - start + 1))
                trials_without_mapping += 1
        
        # Debug: Print mapping statistics
        print(f"\nDebug - Trial mapping statistics:")
        print(f"Trials with mapping: {trials_with_mapping}")
        print(f"Trials without mapping: {trials_without_mapping}")
        
        return valid_indices, trial_indices, original_trial_indices
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Get the actual index from our valid indices list
        start_idx = self.valid_indices[idx]
        X_seq = self.X[start_idx:start_idx + self.sequence_length]
        y_target = self.y[start_idx + self.sequence_length - 1]
        
        # Get the trial index for this sequence
        trial_idx = self.get_original_trial_index(idx)
        
        # Convert to PyTorch tensors with explicit float32 type
        # Return a tuple of (inputs, targets, trial_index)
        return torch.FloatTensor(X_seq), torch.FloatTensor(y_target), torch.tensor(trial_idx, dtype=torch.long)
    
    def get_trial_index(self, idx):
        """Get the filtered trial index for a sequence."""
        if idx < len(self.trial_indices):
            return self.trial_indices[idx]
        return None
    
    def get_original_trial_index(self, idx):
        """Get the original trial index for a sequence."""
        if idx < len(self.original_trial_indices):
            return self.original_trial_indices[idx]
        return None

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

def filter_trial_frames(X, y, sequence_length, trial_size=1400):
    """Filter frames to include context before frame 400."""
    context_frames = sequence_length - 1
    
    # Change these lines to use frames 350-1000
    start_frame = 350  # Changed from 400 - context_frames
    end_frame = 1000   # Keep this the same
    
    filtered_trial_size = end_frame - start_frame + 1  # Should be 651 frames
    
    filtered_X = []
    filtered_y = []
    filtered_to_original = {}  # Maps filtered trial index to original trial index
    
    # Process each trial
    num_trials = len(X) // trial_size
    filtered_trial_idx = 0
    
    for trial_idx in range(num_trials):
        # Extract frames for this trial
        trial_start = trial_idx * trial_size
        trial_end = (trial_idx + 1) * trial_size
        
        # Extract the frames we want to keep (350-1000)
        keep_start = trial_start + start_frame
        keep_end = trial_start + end_frame + 1  # +1 because end is exclusive
        
        # Add to filtered data
        filtered_X.append(X[keep_start:keep_end])
        filtered_y.append(y[keep_start:keep_end])
        
        # Record mapping
        filtered_to_original[filtered_trial_idx] = trial_idx
        filtered_trial_idx += 1
    
    # Concatenate all filtered trials
    filtered_X = np.vstack(filtered_X)
    filtered_y = np.vstack(filtered_y)
    
    print(f"Filtered data shapes:")
    print(f"X: {filtered_X.shape}")
    print(f"y: {filtered_y.shape}")
    print(f"Number of filtered trials: {filtered_trial_idx}")
    print(f"Frames per filtered trial: {filtered_trial_size}")
    
    # DIRECT CHECK: Verify that filtered targets exactly match original targets
    print("\nDIRECT CHECK - Verifying filtered targets match original targets:")
    all_equal = True
    
    # Check each filtered trial
    for filtered_idx in range(filtered_trial_idx):
        original_idx = filtered_to_original[filtered_idx]
        
        # Calculate frame ranges
        original_start = original_idx * trial_size + start_frame
        original_end = original_idx * trial_size + end_frame + 1
        
        filtered_start = filtered_idx * filtered_trial_size
        filtered_end = (filtered_idx + 1) * filtered_trial_size
        
        # Get original and filtered targets for this trial
        original_targets = y[original_start:original_end]
        filtered_targets = filtered_y[filtered_start:filtered_end]
        
        # Check if they are exactly equal
        are_equal = np.array_equal(original_targets, filtered_targets)
        all_equal = all_equal and are_equal
        
        # Only print details for the first few trials
        if filtered_idx < 5:
            print(f"Trial {filtered_idx} (original {original_idx}):")
            print(f"  Original range: {original_start}:{original_end}")
            print(f"  Filtered range: {filtered_start}:{filtered_end}")
            print(f"  Shapes match: {original_targets.shape == filtered_targets.shape}")
            print(f"  Values match: {are_equal}")
            
            if not are_equal:
                # Find the first mismatch
                for i in range(len(original_targets)):
                    if not np.array_equal(original_targets[i], filtered_targets[i]):
                        print(f"  First mismatch at offset {i}:")
                        print(f"    Original: {original_targets[i]}")
                        print(f"    Filtered: {filtered_targets[i]}")
                        print(f"    Difference: {original_targets[i] - filtered_targets[i]}")
                        break
    
    if all_equal:
        print("\nVERIFIED: All filtered targets exactly match the corresponding original targets!")
    else:
        print("\nWARNING: Some filtered targets do not match the original targets!")
    
    return filtered_X, filtered_y, filtered_to_original

def calculate_enhanced_features(trials_data, frames_per_trial):
    """Calculate enhanced features including lagged velocities and moving averages."""
    # Initialize features DataFrame
    features = pd.DataFrame()
    
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
        
        # Add trial features to the main features DataFrame
        features = pd.concat([features, trial_features], ignore_index=True)
    
    # Print column names to help with debugging
    print("\nFeatures DataFrame columns:")
    print(features.columns.tolist())
    
    return features

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

def calculate_psd_features(data, fs=200):
    """
    Calculate power spectral density features for velocity data up to Nyquist frequency.
    
    Args:
        data (numpy.ndarray): Time series data
        fs (int): Sampling frequency in Hz
        
    Returns:
        dict: Dictionary containing PSD features
    """
    # Calculate PSD using Welch's method up to Nyquist frequency (fs/2 = 100 Hz)
    frequencies, psd = signal.welch(data, fs=fs, nperseg=fs)
    
    # Calculate features from PSD
    total_power = np.sum(psd)
    peak_frequency = frequencies[np.argmax(psd)]
    mean_frequency = np.sum(frequencies * psd) / total_power
    
    # Calculate power in different frequency bands
    # More granular frequency bands up to 200 Hz
    bands = {
        'very_low': (0, 10),    # 0-10 Hz: Very slow movements
        'low': (10, 30),        # 10-30 Hz: Slow movements
        'medium': (30, 60),     # 30-60 Hz: Medium speed movements
        'high': (60, 100),      # 60-100 Hz: Fast movements
        'very_high': (100, 200) # 100-200 Hz: Very fast movements/noise
    }
    
    def get_band_power(band):
        mask = (frequencies >= band[0]) & (frequencies < band[1])
        return np.sum(psd[mask])
    
    # Calculate power in each band
    band_powers = {
        f'{band_name}_power': get_band_power(freq_range)
        for band_name, freq_range in bands.items()
    }
    
    # Calculate relative power (percentage of total power in each band)
    relative_powers = {
        f'{band_name}_relative_power': power / total_power
        for band_name, power in band_powers.items()
    }
    
    # Combine all features
    features = {
        'total_power': total_power,
        'peak_freq': peak_frequency,
        'mean_freq': mean_frequency,
        **band_powers,
        **relative_powers
    }
    
    return features

def filter_es_data_by_genotype(df, target_genotype="ES"):
    """
    Filter the ES parquet data to only include rows with genotype 'ES'.
    
    Args:
        df: DataFrame loaded from the parquet file
        target_genotype: The target genotype to filter for (default: 'ES')
        
    Returns:
        Filtered DataFrame
    """
    print(f"Filtering ES data for genotype: {target_genotype}")
    print(f"Original data shape: {df.shape}")
    
    # Check if genotype column exists
    if 'genotype' in df.columns:
        # Check what unique genotypes exist
        unique_genotypes = df['genotype'].unique()
        print(f"Found genotypes in data: {unique_genotypes}")
        
        # Check if our target genotype exists
        if target_genotype in unique_genotypes:
            # Filter to only keep rows with target genotype
            filtered_df = df[df['genotype'] == target_genotype]
            print(f"After filtering for {target_genotype}: {filtered_df.shape} rows")
            return filtered_df
        else:
            print(f"WARNING: {target_genotype} not found in genotypes. Available: {unique_genotypes}")
            
            # If there's only one genotype, assume it's equivalent to ES
            if len(unique_genotypes) == 1:
                print(f"Using the only available genotype: {unique_genotypes[0]}")
                return df
            
            # If there are multiple genotypes but not ES, ask for guidance
            print(f"Multiple genotypes found, but no '{target_genotype}'. Please specify which to use.")
            return df  # Return unfiltered for now with warning
    else:
        print(f"WARNING: No 'genotype' column found in data. Columns: {df.columns.tolist()}")
    
    # If we get here, we couldn't filter by genotype
    print("Could not filter by genotype. Using all data.")
    return df

def add_psd_features(df, base_velocities=None, trial_size=650):
    """Add power spectral density features efficiently to avoid fragmentation."""
    print("Calculating power spectral density features...")
    
    if base_velocities is None:
        base_velocities = ['x_vel', 'y_vel', 'z_vel']
    
    # Initialize new columns
    new_columns = {}
    
    # Initialize PSD feature columns
    for vel in base_velocities:
        # Rename total_power to psd for consistency with input_features
        new_columns[f'{vel}_psd'] = np.zeros(len(df))
        new_columns[f'{vel}_peak_freq'] = np.zeros(len(df))
        new_columns[f'{vel}_mean_freq'] = np.zeros(len(df))
        
        # Initialize band power columns
        bands = {
            'very_low': (0, 10),    # 0-10 Hz: Very slow movements
            'low': (10, 30),        # 10-30 Hz: Slow movements
            'medium': (30, 60),     # 30-60 Hz: Medium speed movements
            'high': (60, 100),      # 60-100 Hz: Fast movements
            'very_high': (100, 200) # 100-200 Hz: Very fast movements/noise
        }
        
        for band_name in bands.keys():
            new_columns[f'{vel}_{band_name}_power'] = np.zeros(len(df))
            new_columns[f'{vel}_{band_name}_relative_power'] = np.zeros(len(df))
    
    # Calculate frames per trial - use the parameter consistently
    frames_per_trial = trial_size
    num_filtered_trials = len(df) // frames_per_trial
    
    # Calculate PSD features for each velocity and trial
    for vel in base_velocities:
        print(f"  Processing {vel}...")
        for trial in range(num_filtered_trials):
            start_idx = trial * frames_per_trial
            end_idx = start_idx + frames_per_trial
            trial_data = df.loc[start_idx:end_idx-1, vel].values
            
            # Calculate PSD using Welch's method
            fs = 200  # 200Hz sampling frequency
            frequencies, psd = signal.welch(trial_data, fs=fs, nperseg=fs)
            
            # Calculate PSD features
            total_power = np.sum(psd)
            peak_frequency = frequencies[np.argmax(psd)]
            mean_frequency = np.sum(frequencies * psd) / total_power if total_power > 0 else 0
            
            # Store results in new columns
            new_columns[f'{vel}_psd'][start_idx:end_idx] = total_power
            new_columns[f'{vel}_peak_freq'][start_idx:end_idx] = peak_frequency
            new_columns[f'{vel}_mean_freq'][start_idx:end_idx] = mean_frequency
            
            # Calculate and store band powers
            bands = {
                'very_low': (0, 10),    # 0-10 Hz: Very slow movements
                'low': (10, 30),        # 10-30 Hz: Slow movements
                'medium': (30, 60),     # 30-60 Hz: Medium speed movements
                'high': (60, 100),      # 60-100 Hz: Fast movements
                'very_high': (100, 200) # 100-200 Hz: Very fast movements/noise
            }
            
            for band_name, (low_freq, high_freq) in bands.items():
                mask = (frequencies >= low_freq) & (frequencies < high_freq)
                band_power = np.sum(psd[mask])
                relative_power = band_power / total_power if total_power > 0 else 0
                
                # Fix: Assign values to the specific trial's range, not the entire column
                new_columns[f'{vel}_{band_name}_power'][start_idx:end_idx] = band_power
                new_columns[f'{vel}_{band_name}_relative_power'][start_idx:end_idx] = relative_power
    
    # Create new DataFrame with PSD features
    new_df = pd.DataFrame(new_columns)
    
    # Concatenate with original DataFrame
    return pd.concat([df, new_df], axis=1)

def prepare_data(data_path, input_features, output_features, sequence_length, output_dir=None, leg_name=None, fixed_trial_splits=None, fixed_filtered_to_original=None):
    """Prepare data for training, validation, and testing."""
    print(f"\nPreparing data from {data_path}...")
    
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
    
    # Check if we need to load ES data as well
    es_data_path = None
    if 'ES' not in data_path:
        # Try to find ES data path
        possible_es_paths = [
            r"Z:\Divya\TEMP_transfers\toAni\4_StopProjectData_forES\df_preproc_fly_centric.parquet",
        ]
        
        for path in possible_es_paths:
            if Path(path).exists():
                es_data_path = path
                print(f"Found ES data at: {es_data_path}")
                break
    
    # Load main data
    if str(data_path).endswith('.csv'):
        df = pd.read_csv(data_path)
    elif str(data_path).endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    print(f"Main data loaded with shape: {df.shape}")
    
    # Load ES data if available
    if es_data_path:
        print(f"Loading ES data from {es_data_path}...")
        es_df = pd.read_parquet(es_data_path)
        
        # Filter to only include ES genotype
        if 'genotype' in es_df.columns:
            es_df = es_df[es_df['genotype'] == 'ES'].copy()
            print(f"ES data filtered to shape: {es_df.shape}")
        
        # Check if the ES data has the required columns
        required_cols = ['x_vel', 'y_vel', 'z_vel'] + output_features
        missing_cols = [col for col in required_cols if col not in es_df.columns]
        
        if missing_cols:
            print(f"Warning: ES data is missing required columns: {missing_cols}")
            print("ES data will not be included.")
        else:
            # Combine the datasets
            print("Combining main data with ES data...")
            df = pd.concat([df, es_df], ignore_index=True)
            print(f"Combined data shape: {df.shape}")
    

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
    
    # Print summary of velocity filtering
    print(f"\nVelocity filtering summary:")
    print(f"Total trials: {len(all_trials)}")
    print(f"Passing trials: {len(passing_trials)}")
    print(f"Failing trials: {len(failing_trials)}")
    
    # Print genotype-specific stats
    genotype_stats = {}
    for _, genotype, _, _ in passing_trials + failing_trials:
        if genotype not in genotype_stats:
            genotype_stats[genotype] = {'total': 0, 'passing': 0}
        genotype_stats[genotype]['total'] += 1
    
    for _, genotype, _, _ in passing_trials:
        genotype_stats[genotype]['passing'] += 1
    
    print("\nGenotype-specific statistics:")
    for genotype, stats in genotype_stats.items():
        print(f"{genotype}:")
        print(f"  Total trials: {stats['total']}")
        print(f"  Passing trials: {stats['passing']}")
        print(f"  Failing trials: {stats['total'] - stats['passing']}")
    
    # If no trials pass the threshold, raise an error
    if len(filtered_trials) == 0:
        raise ValueError("No trials passed the velocity threshold. Check your data or adjust thresholds.")
    
    # Calculate enhanced features for filtered trials
    print("\nCalculating enhanced features for filtered trials...")
    features_df = calculate_enhanced_features(filtered_trials, frames_per_trial=original_trial_size)
    
    # Add PSD features
    print("\nAdding PSD features...")
    features_df = add_psd_features(features_df)
    
    # Filter frames to include context before frame 400
    print("\nFiltering frames to include context before frame 400...")
    
    # Extract input and output features
    X = features_df[input_features].values
    y = features_df[output_features].values
    
    # DIRECT CHECK: Verify that output features match the original data
    print("\nDIRECT CHECK - Verifying output features match original data:")
    all_equal = True
    
    # Check a few trials
    for filtered_idx in range(min(5, len(filtered_trials))):
        # Get the original trial index
        original_idx = velocity_filtered_to_original[filtered_idx]
        
        # Get the original trial data
        original_trial = all_trials[original_idx]
        
        # Get the filtered trial data
        filtered_trial_start = filtered_idx * original_trial_size
        filtered_trial_end = (filtered_idx + 1) * original_trial_size
        
        # For each output feature, compare values
        for feature_idx, feature_name in enumerate(output_features):
            # Get original feature values (frames 350-1000)
            original_values = original_trial.iloc[350:1001][feature_name].values
            
            # Get corresponding values from the features_df
            feature_values = features_df.iloc[filtered_trial_start:filtered_trial_end][feature_name].values
            
            # Check if they are exactly equal
            are_equal = np.array_equal(original_values, feature_values)
            all_equal = all_equal and are_equal
            
            # Only print details for the first feature of each trial
            if feature_idx == 0:
                print(f"Trial {filtered_idx} (original {original_idx}), feature {feature_name}:")
                print(f"  Original values shape: {original_values.shape}")
                print(f"  Feature values shape: {feature_values.shape}")
                print(f"  Values match: {are_equal}")
                
                if not are_equal:
                    # Find the first mismatch
                    for i in range(min(len(original_values), len(feature_values))):
                        if original_values[i] != feature_values[i]:
                            print(f"  First mismatch at offset {i}:")
                            print(f"    Original: {original_values[i]}")
                            print(f"    Feature: {feature_values[i]}")
                            print(f"    Difference: {original_values[i] - feature_values[i]}")
                            break
    
    if all_equal:
        print("\nVERIFIED: All output features match the original data before filtering!")
    else:
        print("\nWARNING: Some output features do not match the original data before filtering!")
        print("This suggests that transformations other than Z-scoring are being applied to the output features.")
    
    # Filter frames
    X_filtered, y_filtered, _ = filter_trial_frames(X, y, sequence_length)  # Ignore the mapping from filter_trial_frames - velocity_filtered_to_original is the correct mapping for original trial indices
    
    # Create stratified train/val/test splits
    train_trials = []
    val_trials = []
    test_trials = []
    
    # Group trials by genotype for stratified sampling
    genotype_trials = {}
    for i, trial_data in enumerate(filtered_trials):
        genotype = trial_data['genotype'].iloc[0]
        if genotype not in genotype_trials:
            genotype_trials[genotype] = []
        genotype_trials[genotype].append((i, trial_data))
    
    # If fixed trial splits are provided, use those instead of creating new ones
    if using_fixed_splits:
        trial_splits = fixed_trial_splits
        # We need to reconstruct the lists of trials based on the fixed split indices
        for idx, trial_data in enumerate(filtered_trials):
            if idx in trial_splits['train']:
                train_trials.append(trial_data)
            elif idx in trial_splits['val']:
                val_trials.append(trial_data)
            elif idx in trial_splits['test']:
                test_trials.append(trial_data)
        
        # Use the provided filtered-to-original mapping
        filtered_to_original = fixed_filtered_to_original
        
        print("\nUsing fixed trial splits:")
        print(f"Train: {len(train_trials)} trials - indices: {sorted(trial_splits['train'])[:10]}...")
        print(f"Validation: {len(val_trials)} trials - indices: {sorted(trial_splits['val'])[:10]}...")
        print(f"Test: {len(test_trials)} trials - indices: {sorted(trial_splits['test'])[:10]}...")
    else:
        # Create stratified splits
        trial_splits = {'train': [], 'val': [], 'test': []}
        
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
                filtered_idx, trial = genotype_trial_data[idx]
                train_trials.append(trial)
                trial_splits['train'].append(filtered_idx)
            
            for idx in val_indices:
                filtered_idx, trial = genotype_trial_data[idx]
                val_trials.append(trial)
                trial_splits['val'].append(filtered_idx)
            
            for idx in test_indices:
                filtered_idx, trial = genotype_trial_data[idx]
                test_trials.append(trial)
                trial_splits['test'].append(filtered_idx)
                
        print("\nGenerated new trial splits:")
        print(f"Train: {len(train_trials)} trials - indices: {sorted(trial_splits['train'])[:10]}...")
        print(f"Validation: {len(val_trials)} trials - indices: {sorted(trial_splits['val'])[:10]}...")
        print(f"Test: {len(test_trials)} trials - indices: {sorted(trial_splits['test'])[:10]}...")
    
    # Create trial masks for train/val/test
    frames_per_trial = 651  # This is the fixed size after filtering
    
    # Calculate how many sequences per trial
    sequences_per_trial = frames_per_trial - sequence_length + 1
    
    # Create masks for each split
    train_mask = np.zeros(len(X_filtered), dtype=bool)
    val_mask = np.zeros(len(X_filtered), dtype=bool)
    test_mask = np.zeros(len(X_filtered), dtype=bool)
    
    # Fill in masks based on trial splits
    for i in range(len(filtered_trials)):
        start_idx = i * frames_per_trial
        end_idx = (i + 1) * frames_per_trial
        
        if i in trial_splits['train']:
            train_mask[start_idx:end_idx] = True
        elif i in trial_splits['val']:
            val_mask[start_idx:end_idx] = True
        elif i in trial_splits['test']:
            test_mask[start_idx:end_idx] = True
    
    # Split data using masks
    X_train = X_filtered[train_mask]
    y_train = y_filtered[train_mask]
    
    X_val = X_filtered[val_mask]
    y_val = y_filtered[val_mask]
    
    X_test = X_filtered[test_mask]
    y_test = y_filtered[test_mask]
    
    print("\nData shapes after splitting:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    
    # DIRECT CHECK: Verify that split targets exactly match filtered targets
    print("\nDIRECT CHECK - Verifying split targets match filtered targets:")
    all_equal = True
    
    # Check each split
    for split_name, (y_split, mask) in zip(['train', 'val', 'test'], 
                                          [(y_train, train_mask), 
                                           (y_val, val_mask), 
                                           (y_test, test_mask)]):
        # Get indices where mask is True
        split_indices = np.where(mask)[0]
        
        # Check if targets match
        targets_match = True
        for i, filtered_idx in enumerate(split_indices[:5]):  # Check first 5 samples
            split_target = y_split[i]
            filtered_target = y_filtered[filtered_idx]
            
            # Check if they are exactly equal
            are_equal = np.array_equal(split_target, filtered_target)
            targets_match = targets_match and are_equal
            all_equal = all_equal and are_equal
            
            print(f"{split_name.capitalize()} sample {i} (filtered index {filtered_idx}):")
            print(f"  Split target: {split_target}")
            print(f"  Filtered target: {filtered_target}")
            print(f"  Equal: {are_equal}")
            
            if not are_equal:
                print(f"  WARNING: Targets don't match!")
                print(f"  Difference: {split_target - filtered_target}")
        
        print(f"All checked {split_name} targets match: {targets_match}")
    
    if all_equal:
        print("\nVERIFIED: All split targets exactly match the corresponding filtered targets!")
    else:
        print("\nWARNING: Some split targets do not match the filtered targets!")
    
    # Scale features
    X_scaler = ZScoreScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_val = X_scaler.transform(X_val)
    X_test = X_scaler.transform(X_test)
    
    # Scale targets
    y_scaler = ZScoreScaler()
    y_train_unscaled = y_train.copy()  # Save unscaled version for comparison
    y_val_unscaled = y_val.copy()
    y_test_unscaled = y_test.copy()
    
    y_train = y_scaler.fit_transform(y_train)
    y_val = y_scaler.transform(y_val)
    y_test = y_scaler.transform(y_test)
    
    # DIRECT CHECK: Verify that only Z-score normalization is applied to output features
    print("\nDIRECT CHECK - Verifying only Z-score normalization is applied to output features:")
    
    # Manually calculate Z-score normalization for a few samples to verify
    print("Checking manual Z-score calculation against the scaler's output:")
    
    # Calculate means and stds manually from the training data
    manual_means = np.mean(y_train_unscaled, axis=0)
    manual_stds = np.std(y_train_unscaled, axis=0)
    manual_stds[manual_stds == 0] = 1  # Handle constant features
    
    # Compare with the scaler's means and stds
    means_match = np.allclose(manual_means, y_scaler.means)
    stds_match = np.allclose(manual_stds, y_scaler.stds)
    
    print(f"Means match: {means_match}")
    print(f"Stds match: {stds_match}")
    
    if not means_match or not stds_match:
        print("WARNING: Scaler's means or stds don't match manual calculation!")
        print(f"Manual means: {manual_means}")
        print(f"Scaler means: {y_scaler.means}")
        print(f"Manual stds: {manual_stds}")
        print(f"Scaler stds: {y_scaler.stds}")
    
    # Check a few samples from each split
    all_equal = True
    for split_name, (y_scaled, y_unscaled) in zip(['train', 'val', 'test'], 
                                                 [(y_train, y_train_unscaled), 
                                                  (y_val, y_val_unscaled), 
                                                  (y_test, y_test_unscaled)]):
        print(f"\nChecking {split_name} split:")
        for i in range(min(5, len(y_scaled))):
            # Get unscaled sample
            unscaled = y_unscaled[i]
            
            # Manually apply Z-score normalization
            manual_scaled = (unscaled - y_scaler.means) / y_scaler.stds
            
            # Get the scaled sample from the scaler
            scaled = y_scaled[i]
            
            # Check if they are equal
            are_equal = np.allclose(manual_scaled, scaled, rtol=1e-5, atol=1e-5)
            all_equal = all_equal and are_equal
            
            print(f"Sample {i}:")
            print(f"  Manual scaled: {manual_scaled}")
            print(f"  Scaler scaled: {scaled}")
            print(f"  Equal: {are_equal}")
            
            if not are_equal:
                print(f"  WARNING: Manual scaling doesn't match scaler's output!")
                print(f"  Difference: {manual_scaled - scaled}")
    
    if all_equal:
        print("\nVERIFIED: Only Z-score normalization is applied to output features!")
    else:
        print("\nWARNING: The transformation applied to output features is not just Z-score normalization!")
    
    # DIRECT CHECK: Verify that Z-score normalization can be correctly reversed
    print("\nDIRECT CHECK - Verifying Z-score normalization can be correctly reversed:")
    all_equal = True
    
    # Check each split
    for split_name, (y_scaled, y_unscaled) in zip(['train', 'val', 'test'], 
                                                 [(y_train, y_train_unscaled), 
                                                  (y_val, y_val_unscaled), 
                                                  (y_test, y_test_unscaled)]):
        print(f"\nChecking {split_name} split:")
        for i in range(min(5, len(y_scaled))):
            # Get scaled and unscaled samples
            scaled = y_scaled[i:i+1]
            unscaled = y_unscaled[i:i+1]
            
            # Inverse transform the scaled sample
            inverse_scaled = y_scaler.inverse_transform(scaled)
            
            # Check if the inverse transformed sample matches the original unscaled sample
            are_equal = np.allclose(unscaled, inverse_scaled, rtol=1e-5, atol=1e-5)
            all_equal = all_equal and are_equal
            
            print(f"Sample {i}:")
            print(f"  Original unscaled: {unscaled.flatten()}")
            print(f"  Scaled: {scaled.flatten()}")
            print(f"  Inverse scaled: {inverse_scaled.flatten()}")
            print(f"  Equal: {are_equal}")
            
            if not are_equal:
                print(f"  WARNING: Original and inverse-scaled don't match!")
                print(f"  Difference: {unscaled.flatten() - inverse_scaled.flatten()}")
                print(f"  Relative difference: {np.abs((unscaled.flatten() - inverse_scaled.flatten()) / (unscaled.flatten() + 1e-10))}")
    
    if all_equal:
        print("\nVERIFIED: Z-score normalization can be correctly reversed!")
    else:
        print("\nWARNING: Z-score normalization cannot be correctly reversed for some samples!")
    
    # Create DataLoaders with the custom TrialSampler for train data
    # For train data, use TrialSampler to shuffle trials but not sequences within trials
    train_dataset = TrialAwareSequenceDataset(
        X_train, y_train, sequence_length, frames_per_trial, velocity_filtered_to_original, trial_splits['train']
    )
    
    # Only use TrialSampler for training data - we want to shuffle trials but keep sequences ordered
    train_sampler = TrialSampler(train_dataset, frames_per_trial, sequence_length)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,
        sampler=train_sampler,  # Use custom sampler instead of shuffle parameter
        num_workers=0
    )
    
    # For validation and test data, we don't need shuffling
    val_dataset = TrialAwareSequenceDataset(
        X_val, y_val, sequence_length, frames_per_trial, velocity_filtered_to_original, trial_splits['val']
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32,
        shuffle=False,  # No shuffling for validation data
        num_workers=0
    )
    
    test_dataset = TrialAwareSequenceDataset(
        X_test, y_test, sequence_length, frames_per_trial, velocity_filtered_to_original, trial_splits['test']
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32,
        shuffle=False,  # No shuffling for test data
        num_workers=0
    )
    
    print("\nDataLoaders created:")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Validation: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")
    print(f"  Batch size: 32")
    print(f"  Using TrialSampler for training data: shuffles trials but keeps sequences ordered")
    
    # Use the velocity-based mapping instead
    return (
        df, filtered_trials, train_trials, val_trials, test_trials,
        train_loader, val_loader, test_loader,
        X_scaler, y_scaler, trial_splits, velocity_filtered_to_original
    )
