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

class R1AnglePredictor(nn.Module):
    def __init__(self, input_size=13, hidden_sizes=[256, 128, 64]):
        super(R1AnglePredictor, self).__init__()
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # Output layer for 8 angles
        layers.append(nn.Linear(prev_size, 8))
        
        self.network = nn.Sequential(*layers)
        
        print("\nModel Architecture:")
        print(f"Input size: {input_size}")
        print("Hidden layers:", hidden_sizes)
        print("Output size: 8 (R1 leg angles)")
        
    def forward(self, x):
        return self.network(x)

def prepare_data(config):
    """Load and prepare data for training."""
    print("\nPreparing data...")
    
    # Load data
    df = pd.read_csv(config['data_path'])
    print(f"Loaded data with shape: {df.shape}")
    
    # Filter for P9LT genotype
    df = df[df['genotype'] == 'P9LT'].copy()
    print(f"Filtered for P9LT: {df.shape}")
    
    # Create trial IDs based on frame numbers
    frame_nums = df['fnum'].values
    trial_ids = []
    current_trial = 0
    last_frame = frame_nums[0]
    
    for frame in frame_nums:
        if frame < last_frame:  # New trial starts when frame number resets
            current_trial += 1
        trial_ids.append(current_trial)
        last_frame = frame
    
    df['trial_id'] = trial_ids
    print(f"Initial number of trials: {len(df['trial_id'].unique())}")
    print(f"Frames per trial: {len(df) / len(df['trial_id'].unique()):.1f}")
    
    # Verify trial lengths and keep only complete trials
    trial_lengths = df.groupby('trial_id').size()
    print("\nTrial length statistics before filtering:")
    print(trial_lengths.describe())
    
    # Keep only trials with exactly 1400 frames
    complete_trials = trial_lengths[trial_lengths == 1400].index
    df = df[df['trial_id'].isin(complete_trials)].copy()
    print(f"\nKeeping only complete 1400-frame trials:")
    print(f"Remaining trials: {len(complete_trials)}")
    print(f"Total frames: {len(df)}")
    
    # Now filter frames 400-1000 from each trial
    filtered_rows = []
    for trial in df['trial_id'].unique():
        trial_data = df[df['trial_id'] == trial]
        filtered_rows.append(trial_data.iloc[400:1000])
    
    df = pd.concat(filtered_rows, axis=0, ignore_index=True)
    print(f"\nFiltered to frames 400-1000:")
    print(f"Number of trials: {len(filtered_rows)}")
    print(f"Total frames: {len(df)} ({len(df) / len(filtered_rows):.1f} frames per trial)")
    
    # Calculate moving averages for velocities within each trial
    base_velocities = ['x_vel', 'y_vel', 'z_vel']
    for window in [5, 10, 20]:
        for vel in base_velocities:
            # Calculate moving average for each trial separately
            ma_values = []
            for trial in df['trial_id'].unique():
                trial_data = df[df['trial_id'] == trial][vel]
                # Calculate moving average and handle edges
                ma = trial_data.rolling(window=window, center=True, min_periods=1).mean()
                ma_values.extend(ma.tolist())
            df[f'{vel}_ma{window}'] = ma_values
    
    # Define input features
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
    
    # Input features: base velocities + R-F-CTr_y
    input_features = ['R-F-CTr_y'] + base_features
    
    # Output features: 8 angles of R1 leg
    output_features = [
        'R1A_flex', 'R1A_rot', 'R1A_abduct',
        'R1B_flex', 'R1B_rot',
        'R1C_flex', 'R1C_rot',
        'R1D_flex'
    ]
    
    print("\nFeature Information:")
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
    
    # Split into train, validation, and test sets by trials
    unique_trials = df['trial_id'].unique()
    train_trials, temp_trials = train_test_split(unique_trials, test_size=0.3, random_state=42)
    val_trials, test_trials = train_test_split(temp_trials, test_size=0.5, random_state=42)
    
    # Create masks for each split
    train_mask = df['trial_id'].isin(train_trials)
    val_mask = df['trial_id'].isin(val_trials)
    test_mask = df['trial_id'].isin(test_trials)
    
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
    
    print("\nDataset Split Information:")
    print(f"Training: {len(train_trials)} trials ({len(X_train)} frames)")
    print(f"Validation: {len(val_trials)} trials ({len(X_val)} frames)")
    print(f"Test: {len(test_trials)} trials ({len(X_test)} frames)")
    print(f"Frames per trial: 600 (frames 400-1000)")
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=config['batch_size']
    )
    
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
        batch_size=config['batch_size']
    )
    
    return (train_loader, val_loader, test_loader), (X_scaler, y_scaler), (input_features, output_features)

def train_model(model, train_loader, val_loader, config):
    """Train the ANN model."""
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
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)  # MAE loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'mae': f'{loss.item():.4f}'})  # Show MAE instead of loss
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Val]')
        
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)  # MAE loss
                val_loss += loss.item()
                val_pbar.set_postfix({'mae': f'{loss.item():.4f}'})  # Show MAE instead of loss
        
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
    
    # Use MAE loss for evaluation
    criterion = nn.L1Loss(reduction='none')
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    
    # Inverse transform predictions and targets
    predictions = y_scaler.inverse_transform(predictions)
    targets = y_scaler.inverse_transform(targets)
    
    # Calculate metrics for each angle
    metrics = {}
    for i, angle in enumerate(output_features):
        # Calculate MAE
        mae = np.mean(np.abs(predictions[:, i] - targets[:, i]))
        
        # Calculate RMSE for comparison
        rmse = np.sqrt(np.mean((predictions[:, i] - targets[:, i])**2))
        
        # Calculate R² score
        r2 = 1 - np.sum((targets[:, i] - predictions[:, i])**2) / np.sum((targets[:, i] - targets[:, i].mean())**2)
        
        metrics[angle] = {
            'mae': mae,
            'rmse': rmse,  # Keep RMSE for comparison
            'r2': r2
        }
        
        # Plot predictions vs actual with MAE in title
        plt.figure(figsize=(10, 6))
        plt.scatter(targets[:, i], predictions[:, i], alpha=0.5)
        plt.plot([targets[:, i].min(), targets[:, i].max()],
                [targets[:, i].min(), targets[:, i].max()],
                'r--', lw=2)
        plt.xlabel(f'Actual {angle}')
        plt.ylabel(f'Predicted {angle}')
        plt.title(f'{angle} Predictions\nMAE = {mae:.3f}, R² = {r2:.3f}')
        plt.grid(True, alpha=0.3)
        
        # Add metrics text box
        plt.text(0.05, 0.95, 
                f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r2:.3f}',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        plt.savefig(plots_dir / f'{angle}_predictions.png')
        plt.close()
    
    # Print summary of MAE values
    print("\nMAE Summary:")
    maes = [m['mae'] for m in metrics.values()]
    print(f"Average MAE across all angles: {np.mean(maes):.3f}")
    print(f"Best MAE: {min(maes):.3f}")
    print(f"Worst MAE: {max(maes):.3f}")
    
    return metrics

def main():
    """Main function to train and evaluate the ANN model."""
    # Configuration
    config = {
        'data_path': "Z:/Divya/TEMP_transfers/toAni/BPN_P9LT_P9RT_flyCoords.csv",
        'batch_size': 64,
        'learning_rate': 1e-4,
        'num_epochs': 200,
        'patience': 20,
        'hidden_sizes': [256, 128, 64]
    }
    
    # Create directories
    results_dir = Path('ann_results')
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
    model = R1AnglePredictor(input_size=len(input_features), hidden_sizes=config['hidden_sizes'])
    
    # Train model
    best_model_state, best_val_loss = train_model(model, train_loader, val_loader, config)
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, y_scaler, output_features, plots_dir)
    
    # Print results
    print("\nFinal Results:")
    for angle, angle_metrics in metrics.items():
        print(f"\n{angle}:")
        for metric, value in angle_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Save model and results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_save_path = models_dir / f'R1_angle_predictor_{timestamp}.pth'
    
    torch.save({
        'model_state_dict': best_model_state,
        'config': config,
        'input_features': input_features,
        'output_features': output_features,
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'metrics': metrics
    }, model_save_path)
    
    print(f"\nModel and results saved to: {model_save_path}")

if __name__ == "__main__":
    main() 