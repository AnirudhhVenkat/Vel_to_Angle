import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
import glob

# Define the ZScoreScaler class needed for model loading
class ZScoreScaler:
    """Z-score normalization scaler"""
    def __init__(self, means=None, stds=None, feature_names=None):
        self.means = means
        self.stds = stds
        self.feature_names = feature_names
    
    def fit(self, X):
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        return self
    
    def transform(self, X):
        return (X - self.means) / self.stds
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        return X * self.stds + self.means

# Define the TCN model architecture
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
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
        self.final = nn.Linear(num_channels[-1], output_size)
        
    def forward(self, x):
        out = self.network(x)
        out = out[:, :, -1]
        return self.final(out)

def process_model(model_path, X_all_scaled, y_trials, y_scaler, input_features, output_features, trial_size, window_size=50):
    """Process a single model and return predictions and metrics"""
    model_path = Path(model_path)
    print(f"\nProcessing model: {model_path}")
    
    try:
        # Try loading with weights_only first
        checkpoint = torch.load(model_path, weights_only=True)
        # Load config from json if available
        config_path = model_path.parent.parent / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {
                'num_channels': [32, 64, 128],
                'kernel_size': 3,
                'dropout': 0.2
            }
    except Exception as e:
        print(f"Warning: Could not load with weights_only=True, falling back to full load: {e}")
        checkpoint = torch.load(model_path)
        config = checkpoint.get('config', {
            'num_channels': [32, 64, 128],
            'kernel_size': 3,
            'dropout': 0.2
        })

    # Create and load model
    model = TemporalConvNet(
        input_size=len(input_features),
        output_size=len(output_features),
        num_channels=config['num_channels'],
        kernel_size=config['kernel_size'],
        dropout=config['dropout'],
        sequence_length=window_size
    )
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Make predictions
    predictions = []
    actual_values = []
    num_trials = len(y_trials)
    
    # Define prediction range
    start_frame = window_size  # Need window_size frames for first prediction
    end_frame = 600  # Only predict up to frame 600
    
    with torch.no_grad():
        for trial in range(num_trials):
            trial_predictions = []
            trial_data = X_all_scaled[trial * trial_size:(trial + 1) * trial_size]
            trial_actual = y_trials[trial, start_frame:end_frame, :]  # Only get actual values up to frame 600
            
            # Only predict frames up to 600
            for i in range(start_frame, end_frame):
                window = trial_data[i - window_size:i]
                window_tensor = torch.FloatTensor(window).T.unsqueeze(0)
                pred = model(window_tensor)
                pred_numpy = pred.numpy()
                pred_original = y_scaler.inverse_transform(pred_numpy)
                trial_predictions.append(pred_original[0])
            
            predictions.append(np.array(trial_predictions))
            actual_values.append(trial_actual)
            print(f"Completed predictions for trial {trial + 1}/{num_trials}")
    
    predictions = np.array(predictions)
    actual_values = np.array(actual_values)
    
    # Calculate metrics only for frames 0-600
    metrics = {}
    for angle_idx, angle_name in enumerate(output_features):
        mae = np.mean(np.abs(predictions[:, :, angle_idx] - actual_values[:, :, angle_idx]))
        rmse = np.sqrt(np.mean((predictions[:, :, angle_idx] - actual_values[:, :, angle_idx]) ** 2))
        metrics[angle_name] = {'mae': mae, 'rmse': rmse}
    
    metrics['overall'] = {
        'mae': np.mean(np.abs(predictions - actual_values)),
        'rmse': np.sqrt(np.mean((predictions - actual_values) ** 2))
    }
    
    return predictions, actual_values, metrics

def create_comparison_plot(predictions, actual_values, output_features, save_path):
    """Create and save comparison plot for a single model"""
    plt.figure(figsize=(20, 25))
    plt.suptitle('Position Comparison - Predicted vs Actual Angles', fontsize=16, y=0.95)
    
    for angle_idx, angle_name in enumerate(output_features):
        plt.subplot(4, 2, angle_idx + 1)
        plt.plot(actual_values[0, :, angle_idx], 'b-', label='Actual', alpha=0.7)
        plt.plot(predictions[0, :, angle_idx], 'r-', label='Predicted', alpha=0.7)
        plt.title(angle_name)
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_prediction_plots(predictions, actual_values, output_features, metrics, output_dir):
    """Create detailed PDF plots showing predictions for all trials.
    
    Args:
        predictions: Array of shape (num_trials, trial_length, num_features)
        actual_values: Array of shape (num_trials, trial_length, num_features)
        output_features: List of feature names
        metrics: Dictionary containing metrics for each feature
        output_dir: Output directory path
    """
    num_trials = predictions.shape[0]
    num_features = len(output_features)
    
    # Create summary plot (first trial) as PNG
    plt.figure(figsize=(20, 25))
    plt.suptitle('Position Comparison - Predicted vs Actual Angles (First Trial)', fontsize=16, y=0.95)
    
    for angle_idx, angle_name in enumerate(output_features):
        plt.subplot(4, 2, angle_idx + 1)
        plt.plot(actual_values[0, :, angle_idx], 'b-', label='Actual', alpha=0.7)
        plt.plot(predictions[0, :, angle_idx], 'r-', label='Predicted', alpha=0.7)
        plt.title(f"{angle_name}\nMAE: {metrics[angle_name]['mae']:.3f}°, R²: {metrics[angle_name]['r2']:.3f}")
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'position_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed PDF with all trials
    with PdfPages(output_dir / 'all_trials_predictions.pdf') as pdf:
        # First page: Summary metrics
        plt.figure(figsize=(12, 8))
        plt.axis('off')
        plt.text(0.1, 0.95, 'Summary Metrics', fontsize=16, fontweight='bold')
        
        y_pos = 0.85
        for angle_name, angle_metrics in metrics.items():
            plt.text(0.1, y_pos, f"\n{angle_name}:", fontsize=12, fontweight='bold')
            y_pos -= 0.05
            plt.text(0.1, y_pos, f"  MAE: {angle_metrics['mae']:.4f}°")
            y_pos -= 0.03
            plt.text(0.1, y_pos, f"  RMSE: {angle_metrics['rmse']:.4f}°")
            y_pos -= 0.03
            plt.text(0.1, y_pos, f"  R²: {angle_metrics['r2']:.4f}")
            y_pos -= 0.05
        
        pdf.savefig()
        plt.close()
        
        # Plot each trial
        for trial in range(num_trials):
            fig = plt.figure(figsize=(20, 25))
            plt.suptitle(f'Trial {trial + 1} of {num_trials}', fontsize=16, y=0.95)
            
            for angle_idx, angle_name in enumerate(output_features):
                plt.subplot(4, 2, angle_idx + 1)
                
                # Get valid indices (non-NaN values)
                valid_mask = ~np.isnan(predictions[trial, :, angle_idx])
                frames = np.arange(len(valid_mask))
                
                # Plot actual values
                plt.plot(frames, actual_values[trial, :, angle_idx], 'b-', 
                        label='Actual', alpha=0.7)
                
                # Plot predicted values (only where valid)
                plt.plot(frames[valid_mask], predictions[trial, valid_mask, angle_idx], 
                        'r-', label='Predicted', alpha=0.7)
                
                # Calculate trial-specific metrics
                pred_valid = predictions[trial, valid_mask, angle_idx]
                actual_valid = actual_values[trial, valid_mask, angle_idx]
                mae = np.mean(np.abs(pred_valid - actual_valid))
                r2 = 1 - np.sum((actual_valid - pred_valid)**2) / np.sum((actual_valid - actual_valid.mean())**2)
                
                plt.title(f"{angle_name}\nTrial MAE: {mae:.3f}°, R²: {r2:.3f}")
                plt.xlabel('Frame')
                plt.ylabel('Angle (degrees)')
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
            
            if trial % 5 == 0:
                print(f"Saved plots for trial {trial + 1}/{num_trials}")

def main():
    # Define model genotypes to test
    model_genotypes = ['P9RT', 
                       'P9LT', 
                       'BPN', 
                       'ALL'
                       ]
    test_genotype = 'ES'  # The genotype we want to test on
    
    # Create base output directory
    output_base = Path('results')
    output_base.mkdir(exist_ok=True, parents=True)
    
    # Load full dataset once
    print("\nLoading dataset...")
    df_full = pd.read_parquet(r"Z:\Divya\TEMP_transfers\toAni\4_StopProjectData_forES\df_preproc_fly_centric.parquet")
    print(f"Full dataset shape: {df_full.shape}")
    print("\nAvailable genotypes in data:")
    print(df_full['genotype'].unique())
    
    # Filter data for test genotype
    df = df_full[df_full['genotype'] == test_genotype].copy()
    if len(df) == 0:
        print(f"No data found for test genotype {test_genotype}, exiting...")
        return
        
    df.reset_index(drop=True, inplace=True)
    print(f"\nFound {len(df)} samples for {test_genotype}")

    trial_size = 1400
    num_trials = len(df) // trial_size
    if num_trials == 0:
        print(f"Not enough data for even one trial (need {trial_size} samples), exiting...")
        return
        
    print(f"Number of complete trials: {num_trials}")
    
    # Verify we have all required columns
    required_base_cols = ['x_vel', 'y_vel', 'z_vel']
    missing_cols = [col for col in required_base_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        print("Cannot proceed...")
        return

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
    
    # Define features
    velocity_features = [
        'x_vel_ma5', 'y_vel_ma5', 'z_vel_ma5',
        'x_vel_ma10', 'y_vel_ma10', 'z_vel_ma10',
        'x_vel_ma20', 'y_vel_ma20', 'z_vel_ma20',
        'x_vel', 'y_vel', 'z_vel'
    ]

    # Define leg configurations to test
    leg_configs = {
        'R-F': {
            'pos_input': 'R-F-FeTi_z',
            'output_features': [
                'R1A_flex', 'R1A_rot', 'R1A_abduct',
                'R1B_flex', 'R1B_rot',
                'R1C_flex', 'R1C_rot',
                'R1D_flex'
            ]
        },
        'L-F': {
            'pos_input': 'L-F-FeTi_z',
            'output_features': [
                'L1A_flex', 'L1A_rot', 'L1A_abduct',
                'L1B_flex', 'L1B_rot',
                'L1C_flex', 'L1C_rot',
                'L1D_flex'
            ]
        },
        'R-M': {
            'pos_input': 'R-M-FeTi_z',
            'output_features': [
                'R2A_flex', 'R2A_rot', 'R2A_abduct',
                'R2B_flex', 'R2B_rot',
                'R2C_flex', 'R2C_rot',
                'R2D_flex'
            ]
        },
        'L-M': {
            'pos_input': 'L-M-FeTi_z',
            'output_features': [
                'L2A_flex', 'L2A_rot', 'L2A_abduct',
                'L2B_flex', 'L2B_rot',
                'L2C_flex', 'L2C_rot',
                'L2D_flex'
            ]
        },
        'R-H': {
            'pos_input': 'R-H-FeTi_z',
            'output_features': [
                'R3A_flex', 'R3A_rot', 'R3A_abduct',
                'R3B_flex', 'R3B_rot',
                'R3C_flex', 'R3C_rot',
                'R3D_flex'
            ]
        },
        'L-H': {
            'pos_input': 'L-H-FeTi_z',
            'output_features': [
                'L3A_flex', 'L3A_rot', 'L3A_abduct',
                'L3B_flex', 'L3B_rot',
                'L3C_flex', 'L3C_rot',
                'L3D_flex'
            ]
        }
    }

    # Process each model genotype
    for model_genotype in model_genotypes:
        print(f"\n{'='*50}")
        print(f"Testing {test_genotype} data with {model_genotype} models")
        print(f"{'='*50}")
        
        # Create output directory for this model genotype
        genotype_dir = output_base / f"{test_genotype}_tested_with_{model_genotype}"
        genotype_dir.mkdir(exist_ok=True)

        # Process each leg configuration
        for leg_name, config in leg_configs.items():
            print(f"\nProcessing {leg_name} leg...")
            
            # Check if all required columns exist
            all_features = velocity_features + [config['pos_input']] + config['output_features']
            missing_cols = [col for col in all_features if col not in df.columns]
            if missing_cols:
                print(f"Missing columns for {leg_name}: {missing_cols}")
                print("Skipping this leg...")
                continue
            
            # Get input features
            input_features = velocity_features + [config['pos_input']]
            output_features = config['output_features']
            
            # Prepare input data
            X_all = df[input_features].values
            y_all = df[output_features].values
            
            if len(X_all) == 0:
                print(f"No data available for {leg_name}, skipping...")
                continue
                
            # Scale input data
            X_scaler = StandardScaler()
            y_scaler = StandardScaler()
            
            # Scale data
            X_all_scaled = X_scaler.fit_transform(X_all)
            y_all_scaled = y_scaler.fit_transform(y_all)
            
            # Reshape into trials
            X_trials = X_all_scaled.reshape(num_trials, trial_size, -1)
            y_trials = y_all_scaled.reshape(num_trials, trial_size, -1)
            
            # Create output directory for this leg
            output_dir = genotype_dir / leg_name
            output_dir.mkdir(exist_ok=True)
            
            # Process model
            model_path = f"C:/Users/bidayelab/Vel_to_Angle/new/tcn_results/{model_genotype}/{leg_name}_leg/best_model.pt"
            if Path(model_path).exists():
                try:
                    # Load model
                    try:
                        # Try loading with weights_only first
                        checkpoint = torch.load(model_path, weights_only=True)
                        # Load config from json if available
                        config_path = Path(model_path).parent.parent / 'config.json'
                        if config_path.exists():
                            with open(config_path, 'r') as f:
                                model_config = json.load(f)
                        else:
                            model_config = {
                                'num_channels': [32, 64, 128],
                                'kernel_size': 3,
                                'dropout': 0.2,
                                'sequence_length': 50
                            }
                    except Exception as e:
                        print(f"Warning: Could not load with weights_only=True, falling back to full load: {e}")
                        checkpoint = torch.load(model_path)
                        model_config = checkpoint.get('config', {
                            'num_channels': [32, 64, 128],
                            'kernel_size': 3,
                            'dropout': 0.2,
                            'sequence_length': 50
                        })

                    # Create and load model
                    model = TemporalConvNet(
                        input_size=len(input_features),
                        output_size=len(output_features),
                        num_channels=model_config['num_channels'],
                        kernel_size=model_config['kernel_size'],
                        dropout=model_config['dropout'],
                        sequence_length=model_config['sequence_length']
                    )
                    
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    model.eval()

                    # Make predictions for each trial
                    predictions = []
                    actual_values = []
                    window_size = model_config['sequence_length']
                    
                    with torch.no_grad():
                        for trial in range(num_trials):
                            trial_predictions = []
                            trial_data = X_trials[trial]
                            trial_actual = y_trials[trial]
                            
                            # Make predictions for each frame in the trial
                            for i in range(window_size, trial_size):
                                window = trial_data[i - window_size:i]
                                window_tensor = torch.FloatTensor(window).T.unsqueeze(0)
                                pred = model(window_tensor)
                                pred_numpy = pred.numpy()
                                pred_original = y_scaler.inverse_transform(pred_numpy)
                                trial_predictions.append(pred_original[0])
                            
                            # Pad the beginning with NaN values for the first window_size frames
                            pad_predictions = np.full((window_size, len(output_features)), np.nan)
                            trial_predictions = np.vstack([pad_predictions, np.array(trial_predictions)])
                            
                            predictions.append(trial_predictions)
                            actual_values.append(y_scaler.inverse_transform(trial_actual))
                            print(f"Completed predictions for trial {trial + 1}/{num_trials}")
                    
                    predictions = np.array(predictions)
                    actual_values = np.array(actual_values)
                    
                    # Calculate metrics for each angle
                    metrics = {}
                    for i, angle_name in enumerate(output_features):
                        # Calculate metrics excluding NaN values
                        valid_mask = ~np.isnan(predictions[:, :, i])
                        pred_valid = predictions[:, :, i][valid_mask]
                        actual_valid = actual_values[:, :, i][valid_mask]
                        
                        mae = np.mean(np.abs(pred_valid - actual_valid))
                        rmse = np.sqrt(np.mean((pred_valid - actual_valid) ** 2))
                        r2 = 1 - np.sum((actual_valid - pred_valid)**2) / np.sum((actual_valid - actual_valid.mean())**2)
                        
                        metrics[angle_name] = {
                            'mae': mae,
                            'rmse': rmse,
                            'r2': r2
                        }
                    
                    # Save metrics
                    metrics_path = output_dir / 'metrics.json'
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics, f, indent=4)
                    
                    # Create detailed PDF plots
                    create_prediction_plots(predictions, actual_values, output_features, metrics, output_dir)
                    
                    # Print metrics
                    print(f"\nMetrics for {leg_name}:")
                    for angle_name, angle_metrics in metrics.items():
                        print(f"\n{angle_name}:")
                        print(f"  MAE: {angle_metrics['mae']:.4f} degrees")
                        print(f"  RMSE: {angle_metrics['rmse']:.4f} degrees")
                        print(f"  R²: {angle_metrics['r2']:.4f}")
                
                except Exception as e:
                    print(f"Error processing model {model_path}: {str(e)}")
                    continue
            else:
                print(f"Model not found: {model_path}")
                continue

        print(f"\nCompleted testing {test_genotype} data with {model_genotype} models")

if __name__ == "__main__":
    main()