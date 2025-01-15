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

def remove_stepcycle_predictions(df, joint_angles, velocity):
    terms = []
    if joint_angles:
        terms.extend(['D_flex', 'ball', '_r', 'cycle', 'tnum', 'fnum', 'flynum', 'SF', 'pos', '_x', '_y', '_z', 'rot', 'abduct'])
    if velocity:
        terms.extend(['D_flex', 'ball', '_r', 'cycle', 'genotype', 'SF', 'pos', '_x', '_y', '_z'])
    
    for term in terms:
        cols = [c for c in df.columns if c.endswith(term)]
        df = df.drop(columns=cols)
    return df

def filter_frames(df, f_0=400, f_f=1000, f_trl=1400):
    return df.loc[((df['fnum'] % f_trl) >= f_0) & ((df['fnum'] % f_trl) < f_f)].copy()

def z_scoring(df):
    for column in df.columns:
        if column not in ['x_vel', 'y_vel', 'z_vel']:
            df[column] = stats.zscore(df[column])
    return df

def create_lagged_features(data, features, lag):
    df = data.copy()
    for feature in features:
        df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        df = df.drop(columns=[f'{feature}'])
    df.dropna(inplace=True)  
    return df

def calculate_accelerations(velocities, chunk_size=600, dt=1):
    num_chunks = velocities.shape[0] // chunk_size
    accelerations = []
    for i in range(num_chunks):
        chunk = velocities[i * chunk_size: (i + 1) * chunk_size]
        accel_chunk = np.diff(chunk, axis=0) / dt
        accelerations.append(accel_chunk)
    return np.vstack(accelerations)

def trim_output_data(output_data, chunk_size=600):
    num_chunks = output_data.shape[0] // chunk_size
    trimmed_output = []
    for i in range(num_chunks):
        chunk = output_data[i * chunk_size: (i + 1) * chunk_size]
        trimmed_chunk = chunk[1:]
        trimmed_output.append(trimmed_chunk)
    return np.vstack(trimmed_output)

def split_into_trials(data, trial_size=600):
    """Split data into trials of fixed size, discarding incomplete trials"""
    num_complete_trials = len(data) // trial_size
    return [data[i*trial_size:(i+1)*trial_size] for i in range(num_complete_trials)]

def create_past_windows_per_trial(trials, targets, window_size, stride=1):
    """Create windows within each trial, ensuring no overlap between trials"""
    X, y = [], []
    
    # Print debug info
    print(f"Number of trials: {len(trials)}")
    print(f"Trial shape: {trials[0].shape if len(trials) > 0 else 'No trials'}")
    
    for trial, target in tqdm(zip(trials, targets), desc="Processing trials", total=len(trials)):
        # Ensure trial length is sufficient
        if len(trial) <= window_size:
            print(f"Warning: Trial length {len(trial)} is <= window size {window_size}")
            continue
            
        # Create windows within this trial
        num_windows = (len(trial) - window_size) // stride
        
        for i in range(num_windows):
            start_idx = i * stride
            end_idx = start_idx + window_size
            
            # Ensure we don't exceed trial boundaries
            if end_idx >= len(trial):
                break
                
            window = trial[start_idx:end_idx]
            target_window = target[end_idx-1]  # Use last frame as target
            
            # Validate shapes
            if window.shape[0] != window_size:
                print(f"Warning: Invalid window shape {window.shape}")
                continue
                
            X.append(window)
            y.append(target_window)
    
    # Convert to arrays and validate
    X = np.array(X)
    y = np.array(y)
    
    print(f"Created {len(X)} windows")
    print(f"Window shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y

def create_past_windows(features, targets, window_size):
    X, y = [], []
    for i in tqdm(range(len(features) - window_size)):
        X.append(features[i:i + window_size])
        y.append(targets[i + window_size])
    return np.array(X), np.array(y)

def prepare_data(config):
    # Load data with error handling
    try:
        # First try with default engine
        df = pd.read_csv(config['data_path'])
    except Exception as e:
        print(f"First attempt to read CSV failed, trying with python engine... Error: {e}")
        try:
            # Try with python engine
            df = pd.read_csv(config['data_path'], engine='python')
        except Exception as e:
            print(f"Second attempt failed. Error: {e}")
            # Try reading with different encoding
            try:
                df = pd.read_csv(config['data_path'], engine='python', encoding='utf-8-sig')
            except Exception as e:
                print(f"Third attempt with utf-8-sig encoding failed. Error: {e}")
                try:
                    df = pd.read_csv(config['data_path'], engine='python', encoding='latin1')
                except Exception as e:
                    print(f"Fourth attempt with latin1 encoding failed. Error: {e}")
                    print(f"All attempts failed. Please check if the file exists and is not corrupted.")
                    print(f"File path: {os.path.abspath(config['data_path'])}")
                    raise e

    print(f"Successfully loaded data with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Remove unwanted columns if they exist
    columns_to_drop = ['genotype']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Reset index without adding it as a column
    df = df.reset_index(drop=True)

    # Process data
    df = filter_frames(df)
    df = df.reset_index(drop=True)
    df = remove_stepcycle_predictions(df, joint_angles=True, velocity=False)
    df = df.drop(columns=['index']) if 'index' in df.columns else df
    df = z_scoring(df)

    # Calculate accelerations
    df['x_acc'] = df['x_vel'].diff().fillna(0)
    df['y_acc'] = df['y_vel'].diff().fillna(0)
    df['z_acc'] = df['z_vel'].diff().fillna(0)

    # Create features and targets
    features = ['x_vel', 'y_vel', 'z_vel', 'x_acc', 'y_acc', 'z_acc']
    X = df[features].values
    y = df.drop(columns=features).values

    # Split data into trials of 600 frames
    trial_size = 600  # Fixed trial size
    X_trials = split_into_trials(X, trial_size)
    y_trials = split_into_trials(y, trial_size)

    if len(X_trials) == 0:
        raise ValueError("No complete trials created. Check data length and trial size.")

    # Create windows within trials
    window_size = config['window_size']
    if window_size >= trial_size:
        raise ValueError(f"Window size ({window_size}) must be less than trial size ({trial_size})")

    X, y = create_past_windows_per_trial(X_trials, y_trials, window_size)

    # Validate resulting data
    if len(X) == 0 or len(y) == 0:
        raise ValueError("No windows created. Check window_size and stride parameters.")

    # Calculate split sizes
    split_ratio = config.get('split_ratio', 0.8)
    total_samples = len(X)
    split_index = int(total_samples * split_ratio)
    test_size = min(config.get('test_size', 7200), total_samples - split_index)

    print(f"Total samples: {total_samples}")
    print(f"Train split: {split_index}")
    print(f"Test size: {test_size}")

    # Split data
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_val = X[split_index:-test_size]
    y_val = y[split_index:-test_size]
    X_test = X[-test_size:]
    y_test = y[-test_size:]

    # Validate split sizes
    for name, data in [("Train", X_train), ("Val", X_val), ("Test", X_test)]:
        print(f"{name} set size: {len(data)}")
        if len(data) == 0:
            raise ValueError(f"Empty {name} set after splitting")

    # Validate data after creating windows
    if len(X) == 0 or len(y) == 0:
        raise ValueError("No data samples generated after creating windows")
        
    # Validate split sizes
    split_index = int(len(X) * split_ratio)
    test_size = min(config['test_size'], len(X) - split_index)
    if split_index <= 0 or test_size <= 0:
        raise ValueError(f"Invalid split sizes. Total samples: {len(X)}, "
                       f"Split index: {split_index}, Test size: {test_size}")

    # Split data with validation
    X_train, X_val = X[:split_index], X[split_index:-test_size]
    y_train, y_val = y[:split_index], y[split_index:-test_size]
    X_test = X[-test_size:]
    y_test = y[-test_size:]

    # Validate tensor creation
    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError(f"Empty dataset after splitting. Train: {len(X_train)}, "
                       f"Val: {len(X_val)}, Test: {len(X_test)}")

    # Create tensors with validation
    try:
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    except Exception as e:
        raise ValueError(f"Failed to create tensors: {str(e)}")

    # Data augmentation
    def augment_batch(X, y):
        if config['augment']:
            noise = torch.randn_like(X) * config['noise_level']
            X = X + noise
        return X, y

    # Create tensors on CPU first
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create datasets with CPU tensors
    train_dataset = TimeSeriesDataset(X_train_tensor, y_train_tensor)
    val_dataset = TimeSeriesDataset(X_val_tensor, y_val_tensor)
    test_dataset = TimeSeriesDataset(X_test_tensor, y_test_tensor)

    # Apply augmentation to training data if enabled
    if config['augment']:
        X_train_tensor, y_train_tensor = augment_batch(X_train_tensor, y_train_tensor)

    # Create data loaders with pin_memory=True
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=0  # Add this to avoid potential multiprocessing issues
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=0
    )

    # Validate data loaders
    if len(train_loader) == 0 or len(val_loader) == 0 or len(test_loader) == 0:
        raise ValueError(f"Empty DataLoader detected. Train: {len(train_loader)}, "
                       f"Val: {len(val_loader)}, Test: {len(test_loader)}")

    return train_loader, val_loader, test_loader