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
from scipy import signal

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
        
        # Get trial information from the data
        self.trial_info = []  # List of (start_idx, end_idx) for each trial
        current_pos = 0
        
        while current_pos < len(self.X):
            # Check if we have enough data for a trial (either 600 or 650 frames)
            if current_pos + 600 <= len(self.X):
                # Check the next chunk of data to determine if it's ES (600) or regular (650)
                if current_pos + 650 <= len(self.X):
                    # Check if the next 50 frames after 600 contain valid data
                    next_chunk = self.X[current_pos + 600:current_pos + 650]
                    if not np.all(np.isnan(next_chunk)):
                        # This is a regular trial (650 frames)
                        self.trial_info.append((current_pos, current_pos + 650))
                        current_pos += 650
                        continue
                
                # This is an ES trial (600 frames) or we're at the end
                self.trial_info.append((current_pos, current_pos + 600))
                current_pos += 600
            else:
                break
        
        print(f"\nSequenceDataset initialized:")
        print(f"Number of trials: {len(self.trial_info)}")
        print(f"Sequence length: {sequence_length}")
        print(f"Total sequences possible: {self.__len__()}")
    
    def __len__(self):
        """Return total number of sequences we can create."""
        total_sequences = 0
        for start_idx, end_idx in self.trial_info:
            trial_length = end_idx - start_idx
            total_sequences += trial_length - self.sequence_length + 1
        return total_sequences
    
    def __getitem__(self, idx):
        """Get a sequence and its target.
        
        Args:
            idx: Index of the sequence to get
        
        Returns:
            sequence: Input sequence of shape (sequence_length, num_features)
            target: Target value for the last frame in sequence
        """
        # Find which trial this index belongs to
        current_idx = idx
        for trial_start, trial_end in self.trial_info:
            trial_length = trial_end - trial_start
            num_sequences = trial_length - self.sequence_length + 1
            
            if current_idx < num_sequences:
                # This sequence belongs to this trial
                sequence_start = trial_start + current_idx
                sequence_end = sequence_start + self.sequence_length
                return (torch.FloatTensor(self.X[sequence_start:sequence_end]),
                       torch.FloatTensor(self.y[sequence_end - 1]))
            
            current_idx -= num_sequences
        
        raise IndexError("Sequence index out of range")

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
    start_frame = 400 - context_frames
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

def calculate_enhanced_features(df, trial_size=1400):
    """Calculate enhanced features including lagged velocities and moving averages."""
    # Initialize features dictionary
    features_dict = {}
    
    # Original velocities
    features_dict['x_vel'] = df['x_vel'].values
    features_dict['y_vel'] = df['y_vel'].values
    features_dict['z_vel'] = df['z_vel'].values
    
    # Calculate number of trials
    num_trials = len(df) // trial_size
    print(f"\nCalculating enhanced features for {num_trials} trials")
    
    # Add lagged velocities (both positive and negative lags)
    lag_values = [1, 2, 3, 5, 10, 20]  # Different lag amounts
    
    # Pre-compute velocity arrays
    vel_arrays = {
        'x': df['x_vel'].values,
        'y': df['y_vel'].values,
        'z': df['z_vel'].values
    }
    
    for lag in lag_values:
        for coord in ['x', 'y', 'z']:
            # Forward lags (future values)
            vel_arr = np.roll(vel_arrays[coord], -lag)
            features_dict[f'{coord}_vel_lag_plus_{lag}'] = vel_arr
            
            # Backward lags (past values)
            vel_arr = np.roll(vel_arrays[coord], lag)
            features_dict[f'{coord}_vel_lag_minus_{lag}'] = vel_arr
    
    # Calculate moving averages
    windows = [5, 10, 20]
    for window in windows:
        for coord in ['x', 'y', 'z']:
            # Velocity moving averages
            ma_vel = pd.Series(vel_arrays[coord]).rolling(window=window, center=True).mean().values
            features_dict[f'{coord}_vel_ma{window}'] = ma_vel
    
    # Calculate derived velocities
    features_dict['velocity_magnitude'] = np.sqrt(
        vel_arrays['x']**2 + vel_arrays['y']**2 + vel_arrays['z']**2
    )
    
    features_dict['xy_velocity'] = np.sqrt(vel_arrays['x']**2 + vel_arrays['y']**2)
    features_dict['xz_velocity'] = np.sqrt(vel_arrays['x']**2 + vel_arrays['z']**2)
    
    # Calculate accelerations using central difference
    dt = 1/200  # 200Hz sampling rate
    for coord in ['x', 'y', 'z']:
        acc = np.zeros_like(vel_arrays[coord])
        acc[1:-1] = (vel_arrays[coord][2:] - vel_arrays[coord][:-2]) / (2 * dt)
        features_dict[f'{coord}_acc'] = acc
    
    # Calculate total acceleration magnitude
    features_dict['acceleration_magnitude'] = np.sqrt(
        features_dict['x_acc']**2 + 
        features_dict['y_acc']**2 + 
        features_dict['z_acc']**2
    )
    
    # Calculate jerk (derivative of acceleration)
    jerk = np.zeros_like(features_dict['acceleration_magnitude'])
    jerk[1:-1] = (features_dict['acceleration_magnitude'][2:] - 
                  features_dict['acceleration_magnitude'][:-2]) / (2 * dt)
    features_dict['jerk_magnitude'] = jerk
    
    # Create DataFrame from dictionary all at once to avoid fragmentation
    features = pd.DataFrame(features_dict)
    
    # Handle NaN values at trial boundaries
    for trial in range(num_trials):
        start_idx = trial * trial_size
        end_idx = (trial + 1) * trial_size
        features.iloc[start_idx:end_idx] = features.iloc[start_idx:end_idx].ffill().bfill()
    
    print(f"Enhanced features shape: {features.shape}")
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
            "C:/Users/bidayelab/Downloads/BPN_P9LT_P9RT_flyCoords.csv"     # Local Windows path
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

def add_psd_features(df, base_velocities=None):
    """Add power spectral density features efficiently to avoid fragmentation."""
    print("Calculating power spectral density features...")
    
    if base_velocities is None:
        base_velocities = ['x_vel', 'y_vel', 'z_vel']
    
    bands = {
        'very_low': (0, 10),    # 0-10 Hz: Very slow movements
        'low': (10, 30),        # 10-30 Hz: Slow movements
        'medium': (30, 60),     # 30-60 Hz: Medium speed movements
        'high': (60, 100),      # 60-100 Hz: Fast movements
        'very_high': (100, 200) # 100-200 Hz: Very fast movements/noise
    }
    
    # Create a dictionary to hold all new columns
    new_columns = {}
    
    # Initialize all new columns at once
    for vel in base_velocities:
        # Basic PSD features
        new_columns[f'{vel}_total_power'] = np.zeros(len(df))
        new_columns[f'{vel}_peak_freq'] = np.zeros(len(df))
        new_columns[f'{vel}_mean_freq'] = np.zeros(len(df))
        
        # Band power features
        for band_name in bands.keys():
            new_columns[f'{vel}_{band_name}_power'] = np.zeros(len(df))
            new_columns[f'{vel}_{band_name}_relative_power'] = np.zeros(len(df))
    
    # Calculate frames per trial
    frames_per_trial = 650  # All trials are now 650 frames
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
            
            # Store features for this trial in our arrays
            new_columns[f'{vel}_total_power'][start_idx:end_idx] = total_power
            new_columns[f'{vel}_peak_freq'][start_idx:end_idx] = peak_frequency
            new_columns[f'{vel}_mean_freq'][start_idx:end_idx] = mean_frequency
            
            # Calculate and store band powers
            for band_name, (low_freq, high_freq) in bands.items():
                mask = (frequencies >= low_freq) & (frequencies < high_freq)
                band_power = np.sum(psd[mask])
                relative_power = band_power / total_power if total_power > 0 else 0
                
                new_columns[f'{vel}_{band_name}_power'][start_idx:end_idx] = band_power
                new_columns[f'{vel}_{band_name}_relative_power'][start_idx:end_idx] = relative_power
    
    # Create a new DataFrame with all the new columns
    new_df = pd.DataFrame(new_columns)
    
    # Join with the original DataFrame efficiently
    return pd.concat([df, new_df], axis=1)

def prepare_data(data_path, input_features, output_features, sequence_length):
    """Load and prepare data for training."""
    print("\nPreparing data...")
    
    # Load regular data
    print("Loading regular data...")
    regular_df = pd.read_csv("Z:/Divya/TEMP_transfers/toAni/BPN_P9LT_P9RT_flyCoords.csv")
    print("Loading ES data...")
    es_df = pd.read_parquet(r"Z:\Divya\TEMP_transfers\toAni\4_StopProjectData_forES\df_preproc_fly_centric.parquet")
    
    # Filter ES data by genotype
    es_df = filter_es_data_by_genotype(es_df, target_genotype="ES")
    
    print(f"Regular data shape: {regular_df.shape}")
    print(f"ES data shape: {es_df.shape}")
    
    # Process regular data
    filtered_trials = []
    trial_size = 1400
    
    # Process regular trials
    num_regular_trials = len(regular_df) // trial_size
    print(f"\nProcessing {num_regular_trials} regular trials...")
    
    for trial in range(num_regular_trials):
        start_idx = trial * trial_size
        end_idx = (trial + 1) * trial_size
        trial_data = regular_df.iloc[start_idx:end_idx]
        genotype = trial_data['genotype'].iloc[0]
        
        # Skip unexpected genotypes
        if genotype not in ['BPN', 'P9RT', 'P9LT']:
            continue
        
        # Get frames 350-1000 for regular genotypes (650 frames)
        frames = trial_data.iloc[350:1000]
        
        # Apply velocity threshold
        if genotype in ['P9RT', 'P9LT']:
            avg_vel = abs(frames['z_vel'].mean())
            threshold = 3
            vel_type = 'z'
        else:  # BPN
            avg_vel = abs(frames['x_vel'].mean())
            threshold = 5
            vel_type = 'x'
        
        # Keep trial if it meets the threshold
        if avg_vel >= threshold:
            filtered_trials.append(frames)
            print(f"Added {genotype} trial with {vel_type}_vel = {avg_vel:.2f}")
    
    # Process ES trials with velocity thresholding
    num_es_trials = len(es_df) // trial_size
    print(f"\nProcessing {num_es_trials} ES trials...")

    for trial in range(num_es_trials):
        start_idx = trial * trial_size
        end_idx = min((trial + 1) * trial_size, len(es_df))
        trial_data = es_df.iloc[start_idx:end_idx]
        
        # Skip incomplete trials
        if len(trial_data) < trial_size:
            print(f"Skipping incomplete trial {trial} with only {len(trial_data)} frames")
            continue
        
        # Get frames 0-650 for ES trials (650 frames)
        frames = trial_data.iloc[0:650]
        
        # Apply velocity threshold for ES data (same as BPN)
        avg_vel = abs(frames['x_vel'].mean())
        threshold = 5  # Using same threshold as BPN
        
        # Keep trial if it meets the threshold
        if avg_vel >= threshold:
            filtered_trials.append(frames)
            print(f"Added ES trial {trial} with x_vel = {avg_vel:.2f}")
    
    if not filtered_trials:
        raise ValueError("No trials remain after velocity filtering!")
    
    # Combine filtered trials
    df = pd.concat(filtered_trials, ignore_index=True)
    print(f"\nShape after combining and filtering: {df.shape}")
    
    # Print genotype distribution
    genotype_counts = df['genotype'].value_counts()
    print("\nGenotype distribution after filtering:")
    for genotype, count in genotype_counts.items():
        frames_per_trial = 650  # All trials are now 650 frames
        num_trials = count // frames_per_trial
        print(f"{genotype}: {num_trials} trials ({count} frames)")
    
    # Verify we have only expected genotypes
    expected_genotypes = {'BPN', 'P9RT', 'P9LT', 'ES'}
    unexpected_genotypes = [g for g in genotype_counts.index if g not in expected_genotypes]
    if unexpected_genotypes:
        raise ValueError(f"Found unexpected genotypes after filtering: {unexpected_genotypes}")
    
    # Calculate moving averages for velocities within each trial
    base_velocities = ['x_vel', 'y_vel', 'z_vel']
    for window in [5, 10, 20]:
        for vel in base_velocities:
            # Create the column first
            df[f'{vel}_ma{window}'] = 0.0
            
            # Calculate moving average for each trial separately
            frames_per_trial = 650  # All trials are now 650 frames
            num_filtered_trials = len(df) // frames_per_trial
            for trial in range(num_filtered_trials):
                start_idx = trial * frames_per_trial
                end_idx = (trial + 1) * frames_per_trial
                trial_data = df.loc[start_idx:end_idx-1, vel]
                # Calculate moving average and handle edges
                ma = trial_data.rolling(window=window, center=True, min_periods=1).mean()
                df.loc[start_idx:end_idx-1, f'{vel}_ma{window}'] = ma.values
    
    # Calculate enhanced features (accelerations, velocity magnitude, acceleration magnitude)
    print("\nCalculating enhanced features...")
    frames_per_trial = 650  # All trials are now 650 frames
    num_filtered_trials = len(df) // frames_per_trial
    
    # Initialize acceleration arrays
    df['x_acc'] = 0.0
    df['y_acc'] = 0.0
    df['z_acc'] = 0.0
    df['velocity_magnitude'] = 0.0
    df['acceleration_magnitude'] = 0.0
    
    # Calculate derivatives and magnitudes for each trial
    for trial in range(num_filtered_trials):
        start_idx = trial * frames_per_trial
        end_idx = start_idx + frames_per_trial
        
        # Calculate accelerations (derivatives of velocity)
        dt = 1/200  # 200Hz sampling rate
        for coord in ['x', 'y', 'z']:
            vel = df.loc[start_idx:end_idx-1, f'{coord}_vel'].values
            acc = np.zeros_like(vel)
            acc[1:-1] = (vel[2:] - vel[:-2]) / (2 * dt)
            df.loc[start_idx:end_idx-1, f'{coord}_acc'] = acc
        
        # Calculate velocity magnitude
        x_vel = df.loc[start_idx:end_idx-1, 'x_vel'].values
        y_vel = df.loc[start_idx:end_idx-1, 'y_vel'].values
        z_vel = df.loc[start_idx:end_idx-1, 'z_vel'].values
        vel_mag = np.sqrt(x_vel**2 + y_vel**2 + z_vel**2)
        df.loc[start_idx:end_idx-1, 'velocity_magnitude'] = vel_mag
        
        # Calculate acceleration magnitude
        x_acc = df.loc[start_idx:end_idx-1, 'x_acc'].values
        y_acc = df.loc[start_idx:end_idx-1, 'y_acc'].values
        z_acc = df.loc[start_idx:end_idx-1, 'z_acc'].values
        acc_mag = np.sqrt(x_acc**2 + y_acc**2 + z_acc**2)
        df.loc[start_idx:end_idx-1, 'acceleration_magnitude'] = acc_mag
    
    # Calculate lagged features
    print("\nCalculating lagged features...")
    lag_values = [5, 10, 20]  # Future lags
    
    # Define velocity-related features for lagging
    velocity_related_features = (
        base_velocities +  # Basic velocities
        [f"{vel}_ma{window}" for vel in base_velocities for window in [5, 10, 20]] +  # Moving averages
        ['x_acc', 'y_acc', 'z_acc'] +  # Accelerations
        ['velocity_magnitude', 'acceleration_magnitude']  # Magnitudes
    )
    
    # Calculate future lags for each feature
    for feature in velocity_related_features:
        for lag in lag_values:
            feature_name = f"{feature}_future_{lag}"
            # Initialize the column first
            df[feature_name] = 0.0
            
            # Calculate for each trial separately
            for trial in range(num_filtered_trials):
                start_idx = trial * frames_per_trial
                end_idx = start_idx + frames_per_trial
                trial_data = df.loc[start_idx:end_idx-1, feature].values
                
                # Future values
                lagged_values = np.zeros_like(trial_data)
                lagged_values[:-lag] = trial_data[lag:]
                lagged_values[-lag:] = trial_data[-1]  # Pad with last value
                
                # Use loc[] instead of chained indexing to avoid warning
                df.loc[start_idx:end_idx-1, feature_name] = lagged_values
    
    # Calculate PSD features
    df = add_psd_features(df, base_velocities)
    
    # At the end of your feature calculation code, after all columns are added, add this line:
    df = df.copy()  # Creates a defragmented copy of the DataFrame
    
    # Define extended input features
    extended_features = [
        # Base features (already included)
        'x_vel_ma5', 'y_vel_ma5', 'z_vel_ma5',
        'x_vel_ma10', 'y_vel_ma10', 'z_vel_ma10',
        'x_vel_ma20', 'y_vel_ma20', 'z_vel_ma20',
        'x_vel', 'y_vel', 'z_vel',
        
        # Enhanced features
        'x_acc', 'y_acc', 'z_acc',
        'velocity_magnitude', 'acceleration_magnitude',
        
        # Lagged features (examples, add more as needed)
        'x_vel_future_5', 'y_vel_future_5', 'z_vel_future_5',
        'x_vel_future_10', 'y_vel_future_10', 'z_vel_future_10',
        'x_vel_future_20', 'y_vel_future_20', 'z_vel_future_20',
        
        # PSD features (examples, add more as needed)
        'x_vel_total_power', 'y_vel_total_power', 'z_vel_total_power',
        'x_vel_peak_freq', 'y_vel_peak_freq', 'z_vel_peak_freq',
        'x_vel_mean_freq', 'y_vel_mean_freq', 'z_vel_mean_freq'
    ]
    
    # Add representative band power features
    for vel in base_velocities:
        for band in ['low', 'medium', 'high']:
            extended_features.append(f'{vel}_{band}_power')
            extended_features.append(f'{vel}_{band}_relative_power')
    
    # Define base features
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
    
    # Use base features as input features by default
    # To use enhanced features, uncomment the next line
    input_features = extended_features
    
    # Print which feature set is being used
    print(f"\nCURRENTLY USING: {'extended features' if input_features == extended_features else 'base features'}")
    print(f"Number of features being used: {len(input_features)}")
    
    # Print the actual features being used
    print("\nACTUAL INPUT FEATURES BEING USED:")
    for i, feat in enumerate(input_features, 1):
        print(f"  {i}. {feat}")
    
    #input_features = base_features
    
    print(f"\nFeature Information:")
    print(f"Base input features ({len(base_features)}):")
    for feat in base_features:
        print(f"  - {feat}")
        if feat in df.columns:
            print(f"    NaN count: {df[feat].isna().sum()}")
    
    print(f"\nExtended features available but not used by default ({len(extended_features)}):")
    for feat in extended_features:
        if feat not in base_features and feat in df.columns:
            print(f"  - {feat}")
    
    print(f"\nOutput features ({len(output_features)}):")
    for feat in output_features:
        print(f"  - {feat}")
    
    # Extract features and targets
    X = df[input_features].values
    y = df[output_features].values
    
    # Calculate split sizes based on filtered trials
    frames_per_trial = 650  # All trials are now 650 frames
    num_filtered_trials = len(df) // frames_per_trial
    train_size = int(0.7 * num_filtered_trials)
    val_size = int(0.15 * num_filtered_trials)
    test_size = num_filtered_trials - train_size - val_size
    
    print(f"\nSplitting data by trials:")
    print(f"Train: {train_size} trials")
    print(f"Validation: {val_size} trials")
    print(f"Test: {test_size} trials")
    
    # Create random permutation of trial indices
    np.random.seed(42)  # For reproducibility
    trial_indices = np.random.permutation(num_filtered_trials)
    
    # Split trial indices into train/val/test
    train_trials = trial_indices[:train_size]
    val_trials = trial_indices[train_size:train_size + val_size]
    test_trials = trial_indices[train_size + val_size:]
    
    print("\nTrial assignments:")
    print(f"Training trials: {sorted(train_trials)}")
    print(f"Validation trials: {sorted(val_trials)}")
    print(f"Test trials: {sorted(test_trials)}")
    
    # Create masks for each split
    train_mask = np.zeros(len(df), dtype=bool)
    val_mask = np.zeros(len(df), dtype=bool)
    test_mask = np.zeros(len(df), dtype=bool)
    
    # Assign trials to splits using the random indices
    for trial in train_trials:
        start_idx = trial * frames_per_trial
        end_idx = (trial + 1) * frames_per_trial
        train_mask[start_idx:end_idx] = True
    
    for trial in val_trials:
        start_idx = trial * frames_per_trial
        end_idx = (trial + 1) * frames_per_trial
        val_mask[start_idx:end_idx] = True
    
    for trial in test_trials:
        start_idx = trial * frames_per_trial
        end_idx = (trial + 1) * frames_per_trial
        test_mask[start_idx:end_idx] = True
    
    # Scale the data
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    # Fit scalers on training data only
    X_train = X_scaler.fit_transform(X[train_mask])
    y_train = y_scaler.fit_transform(y[train_mask])
    
    # Transform validation and test data
    X_val = X_scaler.transform(X[val_mask])
    y_val = y_scaler.transform(y[val_mask])
    X_test = X_scaler.transform(X[test_mask])
    y_test = y_scaler.transform(y[test_mask])
    
    # Create sequence datasets
    train_dataset = SequenceDataset(X_train, y_train, sequence_length)
    val_dataset = SequenceDataset(X_val, y_val, sequence_length)
    test_dataset = SequenceDataset(X_test, y_test, sequence_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    return (train_loader, val_loader, test_loader), (X_scaler, y_scaler), (input_features, output_features), df, test_trials

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

def evaluate_model(model, test_loader, criterion, device, output_features, output_scaler, save_dir, df=None, test_trials=None):
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
    
    # Initialize dictionary to store trial predictions
    trial_predictions_data = {}
    
    # Save predictions and targets in NPZ file
    predictions_dir = save_dir / 'predictions'
    predictions_dir.mkdir(exist_ok=True, parents=True)
    
    # Create genotype array that matches the shape of predictions
    genotype_info = []
    
    # If we have dataframe and test trials info, add genotype info
    if df is not None and test_trials is not None:
        for trial in range(num_trials):
            # Get genotype for this trial
            trial_idx = test_trials[trial]
            start_idx = trial_idx * frames_per_trial
            trial_genotype = df.iloc[start_idx]['genotype']
            genotype_info.append(trial_genotype)
        
        # Save with genotype information
        np.savez(
            predictions_dir / 'trial_predictions.npz',
            predictions=predictions_original,
            targets=targets_original,
            output_features=output_features,
            frames=np.arange(400, 1000),
            genotypes=genotype_info
        )
    else:
        # Save without genotype info if DataFrame not available
        np.savez(
            predictions_dir / 'trial_predictions.npz',
            predictions=predictions_original,
            targets=targets_original,
            output_features=output_features,
            frames=np.arange(400, 1000)
        )
    
    print(f"\nSaved predictions to: {predictions_dir / 'trial_predictions.npz'}")
    
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
    metrics_file = save_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Create plots directory
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
            y_pos -= 0.03
            plt.text(0.1, y_pos, f"  R²: {feature_metrics['r2']:.4f}")
            y_pos -= 0.05
        
        pdf.savefig()
        plt.close()
        
        # Plot each trial
        for trial in range(num_trials):
            # Get genotype for this trial if available
            trial_genotype = "Unknown"
            if df is not None and test_trials is not None and trial < len(test_trials):
                trial_idx = test_trials[trial]
                start_idx = trial_idx * frames_per_trial
                if 'genotype' in df.columns:
                    trial_genotype = df.iloc[start_idx]['genotype']
            
            # Create a figure with subplots for each feature
            fig = plt.figure(figsize=(15, 10))
            plt.suptitle(f'Trial {trial + 1} - Genotype: {trial_genotype}', fontsize=16, y=0.95)
            
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
                
                # Set default frame range
                frame_range = np.arange(400, 400 + len(trial_pred))
                
                # Store trial info
                trial_predictions_data[f'trial_{trial}'] = {
                    'predictions': trial_pred,
                    'targets': trial_target,
                    'frames': frame_range,
                    'genotype': trial_genotype
                }
                
                # Plot predictions and targets
                ax.plot(frame_range, trial_target, 'b-', label='Actual', alpha=0.7)
                ax.plot(frame_range, trial_pred, 'r-', label='Predicted', alpha=0.7)
                
                # Calculate trial-specific metrics
                mae = np.mean(np.abs(trial_pred - trial_target))
                rmse = np.sqrt(np.mean((trial_pred - trial_target)**2))
                r2 = 1 - np.sum((trial_target - trial_pred)**2) / np.sum((trial_target - trial_target.mean())**2)
                
                ax.set_title(f'{feature}\nMAE: {mae:.2f}°, RMSE: {rmse:.2f}°, R²: {r2:.3f}')
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
    """Main function to train and evaluate LSTM models for all 6 legs."""
    # Configuration
    base_config = {
        'sequence_length': 50,
        'hidden_size': 256,
        'num_layers': 2,
        'dropout': 0.2,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 200,
        'patience': 20
    }
    
    # Create base directories
    results_dir = Path('lstm_results')
    results_dir.mkdir(exist_ok=True, parents=True)
    print("\nCreated base results directory:", results_dir)
    
    # Create error log file
    error_log = results_dir / 'error_log.txt'
    with open(error_log, 'w') as f:
        f.write(f"Training started at {datetime.now()}\n")
        f.write("-" * 80 + "\n\n")
    
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
    
    # Train model for each leg
    for leg_name, leg_info in legs.items():
        try:
            print(f"\n{'='*80}")
            print(f"Training model for {leg_name} leg")
            print(f"{'='*80}")
            
            # Create leg-specific directories
            leg_dir = results_dir / leg_name
            leg_models_dir = leg_dir / 'models'
            leg_plots_dir = leg_dir / 'plots'
            
            leg_dir.mkdir(exist_ok=True, parents=True)
            leg_models_dir.mkdir(exist_ok=True, parents=True)
            leg_plots_dir.mkdir(exist_ok=True, parents=True)
            
            print(f"\nCreated directory structure for {leg_name}:")
            print(f"- {leg_dir}")
            print(f"- {leg_models_dir}")
            print(f"- {leg_plots_dir}")
            
            # Get data path
            data_path = get_available_data_path(leg_name)
            
            # Use leg-specific output features
            output_features = leg_info['angles']
            
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
                'z_vel',
                'x_acc_ma5',    # Acceleration
                'y_acc_ma5',
                'z_acc_ma5',
                'x_acc_ma10',
                'y_acc_ma10',
                'z_acc_ma10',
                'x_acc_ma20',
                'y_acc_ma20',
                'z_acc_ma20',
                'x_acc',        # Raw acceleration
                'y_acc',
                'z_acc',
                'vel_mag_ma5',  # Velocity magnitude
                'vel_mag_ma10',
                'vel_mag_ma20',
                'vel_mag',      # Raw velocity magnitude
                'acc_mag_ma5',  # Acceleration magnitude
                'acc_mag_ma10',
                'acc_mag_ma20',
                'acc_mag',      # Raw acceleration magnitude
                'x_vel_lag1',   # Lagged features
                'y_vel_lag1',
                'z_vel_lag1',
                'x_vel_lag2',
                'y_vel_lag2',
                'z_vel_lag2',
                'x_vel_lag3',
                'y_vel_lag3',
                'z_vel_lag3',
                'x_vel_lag4',
                'y_vel_lag4',
                'z_vel_lag4',
                'x_vel_lag5',
                'y_vel_lag5',
                'z_vel_lag5',
                'x_vel_psd',    # Power Spectral Density features
                'y_vel_psd',
                'z_vel_psd'
            ]
            
            print(f"\nFeatures for {leg_name}:")
            print(f"Input features ({len(input_features)}):")
            for feat in input_features:
                print(f"  - {feat}")
            print(f"\nOutput features ({len(output_features)}):")
            for feat in output_features:
                print(f"  - {feat}")
            
            # Prepare data for this leg
            (train_loader, val_loader, test_loader), (X_scaler, y_scaler), (input_features, output_features), df, test_trials = prepare_data(
                data_path,
                input_features,
                output_features,
                base_config['sequence_length']
            )
            
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
            
            print(f"\nTraining completed for {leg_name}. Best validation loss: {best_val_loss:.4f}")
            
            # Load best model
            model.load_state_dict(best_model_state)
            
            # Evaluate model
            metrics = evaluate_model(
                model,
                test_loader,
                criterion,
                device,
                output_features,
                y_scaler,
                leg_plots_dir,
                df=df,
                test_trials=test_trials
            )
            
            # Save model and results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_save_path = leg_models_dir / f'{leg_name}_lstm_{timestamp}.pth'
            
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