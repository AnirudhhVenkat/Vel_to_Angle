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

def prepare_data(config):
    try:
        # Load only necessary columns
        df_sample = pd.read_csv(config['data_path'], nrows=1)
        
        # Identify relevant columns
        velocity_cols = ['x_vel', 'y_vel', 'z_vel']
        joint_cols = [col for col in df_sample.columns 
                     if col.endswith(('_flex', '_rot', '_abduct'))]
        frame_cols = ['fnum']
        
        usecols = velocity_cols + joint_cols + frame_cols
        
        # Load data efficiently
        df = pd.read_csv(
            config['data_path'],
            usecols=usecols,
            dtype={col: np.float32 for col in usecols if col != 'fnum'},
            engine='c'
        )
        
        # Filter frames
        df = filter_frames(df)
        
        # Calculate accelerations using numpy operations
        velocities = df[velocity_cols].values
        accelerations = np.diff(velocities, axis=0, prepend=velocities[0:1])
        
        # Combine features and standardize
        features = np.concatenate([velocities, accelerations], axis=1)
        features = stats.zscore(features, axis=0)
        targets = stats.zscore(df[joint_cols].values, axis=0)
        
        # Split into trials
        trial_size = 600
        num_trials = len(features) // trial_size
        
        X_trials = [features[i*trial_size:(i+1)*trial_size] for i in range(num_trials)]
        y_trials = [targets[i*trial_size:(i+1)*trial_size] for i in range(num_trials)]
        
        # Create windows
        window_size = config['window_size']
        X, y = create_past_windows_per_trial(X_trials, y_trials, window_size)
        
        # Split data
        split_ratio = config.get('split_ratio', 0.8)
        total_samples = len(X)
        split_index = int(total_samples * split_ratio)
        test_size = min(config.get('test_size', 7200), total_samples - split_index)
        
        # Create train/val/test splits
        X_train = torch.tensor(X[:split_index], dtype=torch.float32)
        y_train = torch.tensor(y[:split_index], dtype=torch.float32)
        X_val = torch.tensor(X[split_index:-test_size], dtype=torch.float32)
        y_val = torch.tensor(y[split_index:-test_size], dtype=torch.float32)
        X_test = torch.tensor(X[-test_size:], dtype=torch.float32)
        y_test = torch.tensor(y[-test_size:], dtype=torch.float32)
        
        # Data augmentation if enabled
        if config['augment']:
            noise = torch.randn_like(X_train) * config['noise_level']
            X_train = X_train + noise
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        test_dataset = TimeSeriesDataset(X_test, y_test)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=0
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=0
        )
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"Error in prepare_data: {str(e)}")
        raise