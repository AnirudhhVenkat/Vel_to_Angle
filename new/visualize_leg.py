import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import torch
import torch.nn as nn
from pathlib import Path
import json
import argparse
import pandas as pd
import glob
from datetime import datetime

class Chomp1d(nn.Module):
    """Helper module to ensure causal convolutions"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    """TCN block with dilated causal convolutions"""
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
    """Temporal Convolutional Network for leg angle prediction"""
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

class ZScoreScaler:
    """Scaler for z-score normalization with per-feature parameters"""
    def __init__(self, means, stds, feature_names):
        self.means = means
        self.stds = stds
        self.feature_names = feature_names
    
    def transform(self, X):
        X_scaled = np.zeros_like(X)
        for i, feature in enumerate(self.feature_names):
            X_scaled[:, i] = (X[:, i] - self.means[feature]) / self.stds[feature]
        return X_scaled
    
    def inverse_transform(self, X):
        X_inv = np.zeros_like(X)
        for i, feature in enumerate(self.feature_names):
            X_inv[:, i] = X[:, i] * self.stds[feature] + self.means[feature]
        return X_inv

def create_leg_segments(angles, leg_prefix):
    """Create leg segments for visualization based on joint angles.
    
    Args:
        angles: Dictionary containing joint angles for one frame
        leg_prefix: Prefix indicating which leg (e.g., 'R-F', 'L-F', etc.)
    
    Returns:
        numpy array of shape (5, 3) containing 3D coordinates of leg points
    """
    # Initialize points array with the correct shape
    points = np.zeros((5, 3))  # 5 points (joints), 3 coordinates (x,y,z)
    
    # Set base x-offset based on leg side (left/right)
    base_x_offset = 3.0 if leg_prefix.startswith('L') else -3.0
    x_offset = base_x_offset
    y_offset = 0.0
    
    # Set y-offset based on leg position (front/mid/hind)
    if 'F' in leg_prefix or '1' in leg_prefix:  # Front legs
        y_offset = 2.0     # Align with FRONT label
        x_offset = base_x_offset
    elif 'M' in leg_prefix or '2' in leg_prefix:  # Middle legs
        y_offset = 0.0     # Align with MIDDLE label
        x_offset = base_x_offset * 0.9  # Bring middle legs slightly inward
    elif 'H' in leg_prefix or '3' in leg_prefix:  # Hind legs
        y_offset = -2.0    # Align with HIND label
        x_offset = base_x_offset * 0.8  # Bring hind legs more inward
    
    # Extract angles for this leg
    if '-' in leg_prefix:  # New style prefixes (R-F, L-F, etc.)
        prefix = leg_prefix.replace('-', '')  # Remove hyphen for angle lookup
    else:  # Old style prefixes (R1, L1, etc.)
        prefix = leg_prefix
    
    # Origin point (thorax-coxa joint)
    points[0] = np.array([x_offset, y_offset, 0])
    
    # Get the angles we need - handle both array and scalar inputs
    def get_angle(name):
        angle = angles[f'{prefix}{name}']
        if isinstance(angle, (np.ndarray, list)):
            if len(angle) == 1:
                return float(angle[0])
            else:
                return float(angle)  # Take first value if array
        return float(angle)
    
    try:
        # Get angles and convert to radians
        A_flex = np.radians(get_angle('A_flex'))
        A_rot = np.radians(get_angle('A_rot'))
        A_abduct = np.radians(get_angle('A_abduct'))
        B_flex = np.radians(get_angle('B_flex'))
        B_rot = np.radians(get_angle('B_rot'))
        C_flex = np.radians(get_angle('C_flex'))
        C_rot = np.radians(get_angle('C_rot'))
        D_flex = np.radians(get_angle('D_flex'))
    except Exception as e:
        print(f"Error extracting angles for {prefix}: {e}")
        print(f"Available angles: {list(angles.keys())}")
        print(f"Angle values: {angles}")
        raise
    
    # Segment lengths (can be adjusted)
    coxa_length = 0.5
    femur_length = 1.0
    tibia_length = 1.0
    tarsus_length = 0.5
    
    # Calculate joint positions using proper rotation order
    # Coxa-femur joint (point 1)
    R_A = np.array([
        [np.cos(A_rot), -np.sin(A_rot), 0],
        [np.sin(A_rot), np.cos(A_rot), 0],
        [0, 0, 1]
    ])
    
    R_flex = np.array([
        [np.cos(A_flex), 0, -np.sin(A_flex)],
        [0, 1, 0],
        [np.sin(A_flex), 0, np.cos(A_flex)]
    ])
    
    R_abduct = np.array([
        [1, 0, 0],
        [0, np.cos(A_abduct), -np.sin(A_abduct)],
        [0, np.sin(A_abduct), np.cos(A_abduct)]
    ])
    
    # Combined rotation matrix for first joint
    R_A_combined = R_A @ R_flex @ R_abduct
    points[1] = points[0] + (R_A_combined @ np.array([coxa_length, 0, 0]))
    
    # Femur-tibia joint (point 2)
    R_B = np.array([
        [np.cos(B_rot), -np.sin(B_rot), 0],
        [np.sin(B_rot), np.cos(B_rot), 0],
        [0, 0, 1]
    ])
    
    R_B_flex = np.array([
        [np.cos(B_flex), 0, -np.sin(B_flex)],
        [0, 1, 0],
        [np.sin(B_flex), 0, np.cos(B_flex)]
    ])
    
    R_B_combined = R_B @ R_B_flex
    points[2] = points[1] + (R_B_combined @ np.array([femur_length, 0, 0]))
    
    # Tibia-tarsus joint (point 3)
    R_C = np.array([
        [np.cos(C_rot), -np.sin(C_rot), 0],
        [np.sin(C_rot), np.cos(C_rot), 0],
        [0, 0, 1]
    ])
    
    R_C_flex = np.array([
        [np.cos(C_flex), 0, -np.sin(C_flex)],
        [0, 1, 0],
        [np.sin(C_flex), 0, np.cos(C_flex)]
    ])
    
    R_C_combined = R_C @ R_C_flex
    points[3] = points[2] + (R_C_combined @ np.array([tibia_length, 0, 0]))
    
    # Tarsus tip (point 4)
    R_D = np.array([
        [np.cos(D_flex), 0, -np.sin(D_flex)],
        [0, 1, 0],
        [np.sin(D_flex), 0, np.cos(D_flex)]
    ])
    
    points[4] = points[3] + (R_D @ np.array([tarsus_length, 0, 0]))
    
    return points

def update_plot(frame, fig, ax, actual_angles, pred_angles, leg_prefix):
    ax.cla()
    
    # Recreate basic plot setup
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_zlabel('Z', fontsize=14)
    ax.set_title(f'Leg Movement Animation - {leg_prefix}', fontsize=16, pad=20)
    
    # Set consistent axis limits
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    ax.set_zlim([1.5, -1.5])
    
    # Add grid for better depth perception
    ax.grid(True, alpha=0.3)
    
    # Set viewing angle with slower rotation
    ax.view_init(elev=20, azim=frame % 360)
    
    # Add position labels in red with better positioning
    # Side labels
    if leg_prefix.startswith('L'):
        ax.text(2.0, 0, 0, 'LEFT', color='red', fontsize=14, fontweight='bold')
        ax.text(-2.0, 0, 0, 'RIGHT', color='red', fontsize=14, fontweight='bold', alpha=0.3)
    else:
        ax.text(2.0, 0, 0, 'LEFT', color='red', fontsize=14, fontweight='bold', alpha=0.3)
        ax.text(-2.0, 0, 0, 'RIGHT', color='red', fontsize=14, fontweight='bold')
    
    # Position labels with better visibility
    ax.text(0, 2.0, 0, 'FRONT', color='red', fontsize=14, ha='center')
    ax.text(0, 0.0, 0, 'MIDDLE', color='red', fontsize=14, ha='center')
    ax.text(0, -2.0, 0, 'HIND', color='red', fontsize=14, ha='center')
    
    # Get current frame angles
    actual_angles_frame = {k: v[frame] for k, v in actual_angles.items()}
    pred_angles_frame = {k: v[frame] for k, v in pred_angles.items()}
    
    # Create leg segments
    actual_points = create_leg_segments(actual_angles_frame, leg_prefix)
    pred_points = create_leg_segments(pred_angles_frame, leg_prefix)
    
    # Plot actual trajectory (blue)
    actual_points = np.array(actual_points)
    ax.plot(actual_points[:, 0], actual_points[:, 1], actual_points[:, 2], 
            'b-', linewidth=3, label='Actual')
    ax.scatter(actual_points[:, 0], actual_points[:, 1], actual_points[:, 2], 
              c='blue', marker='o')
    
    # Plot predicted trajectory (red)
    pred_points = np.array(pred_points)
    ax.plot(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
            'r-', linewidth=3, label='Predicted')
    ax.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
              c='red', marker='o')
    
    # Add legend with better positioning
    ax.legend(loc='upper right', fontsize=12, bbox_to_anchor=(1.15, 1))
    
    # Adjust tick label size
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    return fig,

def visualize_leg_movement(actual_angles, pred_angles, save_path, interval=50):
    """Visualize the movement of all legs simultaneously."""
    print("\nCreating visualization...")
    
    # Calculate actual fps based on interval
    fps = min(30, 1000 // interval)  # Cap at 30 fps, scale down based on interval
    
    # Create figure
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set wider axis limits to accommodate all legs
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([2, -2])
    
    # Add grid for better depth perception
    ax.grid(True, alpha=0.3)
    
    # Set initial viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Add position labels with corrected positioning and better visibility
    # LEFT/RIGHT labels - placed on the correct sides
    ax.text(-3.5, 0, 0, 'LEFT', color='black', fontsize=14, fontweight='bold', ha='right')
    ax.text(3.5, 0, 0, 'RIGHT', color='black', fontsize=14, fontweight='bold', ha='left')
    
    # FRONT/MIDDLE/HIND labels - aligned with actual leg positions
    ax.text(0, 2.0, 0, 'FRONT', color='black', fontsize=14, fontweight='bold', ha='center')
    ax.text(0, 0.0, 0, 'MIDDLE', color='black', fontsize=14, fontweight='bold', ha='center')
    ax.text(0, -2.0, 0, 'HIND', color='black', fontsize=14, fontweight='bold', ha='center')
    
    # Add background boxes to make labels more visible
    for text in ax.texts:
        text.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Set title
    ax.set_title('All Legs Movement Animation\nBlue: Actual, Red: Predicted', fontsize=16, pad=20)
    
    # Add axis labels
    ax.set_xlabel('X (Left/Right)', fontsize=12)
    ax.set_ylabel('Y (Front/Back)', fontsize=12)
    ax.set_zlim([2, -2])  # Invert Z axis to match biological orientation
    ax.set_zlabel('Z (Up/Down)', fontsize=12)
    
    # Get all unique leg prefixes from the angle names
    leg_prefixes = set()
    for angle_name in actual_angles.keys():
        if '-' in angle_name:  # New style (R-F, L-F)
            prefix = angle_name.split('_')[0]  # Get R-F, L-F, etc.
        else:  # Old style (R1, L1)
            prefix = angle_name[:2]  # Get R1, L1, etc.
        leg_prefixes.add(prefix)
    
    # Initialize empty lists to store line objects for animation
    lines_pred = []
    points_pred = []
    lines_actual = []
    points_actual = []
    
    # Create initial plots for each leg
    for leg_prefix in leg_prefixes:
        # Get angles for this leg
        leg_angles = [name for name in actual_angles.keys() if name.startswith(leg_prefix.replace('-', ''))]
        
        # Create dictionaries for current frame angles
        leg_actual = {name: actual_angles[name][0] for name in leg_angles}
        leg_pred = {name: pred_angles[name][0] for name in leg_angles}
        
        # Create leg segments
        actual_points = create_leg_segments(leg_actual, leg_prefix)
        pred_points = create_leg_segments(leg_pred, leg_prefix)
        
        # Plot predicted trajectory (red)
        line_p, = ax.plot(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                         'r-', linewidth=2, label='Predicted' if len(lines_pred) == 0 else "")
        point_p, = ax.plot(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                          'ro', markersize=4)
        
        # Plot actual trajectory (blue)
        line_a, = ax.plot(actual_points[:, 0], actual_points[:, 1], actual_points[:, 2], 
                         'b-', linewidth=2, label='Actual' if len(lines_actual) == 0 else "")
        point_a, = ax.plot(actual_points[:, 0], actual_points[:, 1], actual_points[:, 2], 
                          'bo', markersize=4)
        
        # Store line objects
        lines_pred.append(line_p)
        points_pred.append(point_p)
        lines_actual.append(line_a)
        points_actual.append(point_a)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=12, bbox_to_anchor=(1.15, 1))
    
    def update_plot(frame):
        for i, leg_prefix in enumerate(leg_prefixes):
            # Get angles for this leg
            leg_angles = [name for name in actual_angles.keys() if name.startswith(leg_prefix.replace('-', ''))]
            
            # Create dictionaries for current frame angles
            leg_actual = {name: actual_angles[name][frame] for name in leg_angles}
            leg_pred = {name: pred_angles[name][frame] for name in leg_angles}
            
            try:
                # Create leg segments
                actual_points = create_leg_segments(leg_actual, leg_prefix)
                pred_points = create_leg_segments(leg_pred, leg_prefix)
                
                # Update predicted trajectory
                lines_pred[i].set_data(pred_points[:, 0], pred_points[:, 1])
                lines_pred[i].set_3d_properties(pred_points[:, 2])
                points_pred[i].set_data(pred_points[:, 0], pred_points[:, 1])
                points_pred[i].set_3d_properties(pred_points[:, 2])
                
                # Update actual trajectory
                lines_actual[i].set_data(actual_points[:, 0], actual_points[:, 1])
                lines_actual[i].set_3d_properties(actual_points[:, 2])
                points_actual[i].set_data(actual_points[:, 0], actual_points[:, 1])
                points_actual[i].set_3d_properties(actual_points[:, 2])
            except Exception as e:
                print(f"Error updating leg {leg_prefix} at frame {frame}: {e}")
                continue
        
        # Update viewing angle for rotation (slower rotation)
        ax.view_init(elev=20, azim=(frame % 720) / 2)  # Slower rotation by dividing by 2
        return lines_pred + points_pred + lines_actual + points_actual
    
    # Create animation with adjusted frame rate
    n_frames = len(next(iter(actual_angles.values())))
    anim = animation.FuncAnimation(
        fig, update_plot, frames=n_frames,
        interval=interval, blit=True
    )
    
    # Try multiple approaches to save the animation
    if save_path:
        print(f"Saving animation to {save_path}...")
        try:
            # Try FFmpeg first with adjusted fps
            writer = animation.FFMpegWriter(fps=fps)
            anim.save(save_path, writer=writer)
            print("Successfully saved with FFMpegWriter")
        except Exception as e1:
            print(f"Error saving with FFMpegWriter: {str(e1)}")
            try:
                # Try ImageMagick second
                writer = animation.ImageMagickWriter(fps=fps)
                anim.save(save_path, writer=writer)
                print("Successfully saved with ImageMagickWriter")
            except Exception as e2:
                print(f"Error saving with ImageMagickWriter: {str(e2)}")
                try:
                    # Try Pillow as last resort
                    writer = animation.PillowWriter(fps=fps)
                    anim.save(save_path, writer=writer)
                    print("Successfully saved with PillowWriter")
                except Exception as e3:
                    print(f"Error saving with PillowWriter: {str(e3)}")
                    print("Failed to save animation with any available writer")
    
    return fig, anim

def get_test_trial_indices(model_path):
    """Get the indices of test trials from the model's configuration"""
    results = torch.load(model_path)
    config = results['config']
    
    # Load and preprocess data
    df = pd.read_csv(config['data_path'])
    df = df[df['genotype'] == config['genotype']]
    
    # Get trial indices
    trial_size = 1400
    num_trials = len(df) // trial_size
    
    # Create trial indices and shuffle with same seed
    trial_indices = np.arange(num_trials)
    np.random.seed(42)  # Same seed as in training
    np.random.shuffle(trial_indices)
    
    # Get test indices (last 15% of trials)
    train_size = int(0.7 * num_trials)
    val_size = int(0.15 * num_trials)
    test_indices = trial_indices[train_size + val_size:]
    
    return sorted(test_indices)

def get_leg_angles(leg_prefix):
    """Get the angle names for a specific leg"""
    # Map leg prefix to angle names
    leg_mapping = {
        'R-F': 'R1', 'L-F': 'L1',
        'R-M': 'R2', 'L-M': 'L2',
        'R-H': 'R3', 'L-H': 'L3'
    }
    
    # Convert new format to old format
    old_prefix = leg_mapping.get(leg_prefix, leg_prefix)
    
    if old_prefix.startswith('R1'):
        return ['R1A_flex', 'R1A_rot', 'R1A_abduct', 'R1B_flex', 
                'R1B_rot', 'R1C_flex', 'R1C_rot', 'R1D_flex']
    elif old_prefix.startswith('L1'):
        return ['L1A_flex', 'L1A_rot', 'L1A_abduct', 'L1B_flex', 
                'L1B_rot', 'L1C_flex', 'L1C_rot', 'L1D_flex']
    elif old_prefix.startswith('R2'):
        return ['R2A_flex', 'R2A_rot', 'R2A_abduct', 'R2B_flex', 
                'R2B_rot', 'R2C_flex', 'R2C_rot', 'R2D_flex']
    elif old_prefix.startswith('L2'):
        return ['L2A_flex', 'L2A_rot', 'L2A_abduct', 'L2B_flex', 
                'L2B_rot', 'L2C_flex', 'L2C_rot', 'L2D_flex']
    elif old_prefix.startswith('R3'):
        return ['R3A_flex', 'R3A_rot', 'R3A_abduct', 'R3B_flex', 
                'R3B_rot', 'R3C_flex', 'R3C_rot', 'R3D_flex']
    elif old_prefix.startswith('L3'):
        return ['L3A_flex', 'L3A_rot', 'L3A_abduct', 'L3B_flex', 
                'L3B_rot', 'L3C_flex', 'L3C_rot', 'L3D_flex']
    else:
        raise ValueError(f"Invalid leg prefix: {leg_prefix} (mapped to {old_prefix})")

def load_predicted_angles(model_path, trial_id=None, leg=None):
    """
    Load model and generate predictions for test trials
    """
    # Load model and config safely
    try:
        # First try loading with weights_only=True
        state_dict = torch.load(model_path, weights_only=True)
        # Load config and scalers separately
        with open(model_path.parent / 'config.json', 'r') as f:
            config = json.load(f)
        with open(model_path.parent / 'scalers.pkl', 'rb') as f:
            scalers = torch.load(f)
            X_scaler = scalers['X_scaler']
            y_scaler = scalers['y_scaler']
    except:
        print("Warning: Could not load model components separately. Falling back to unsafe loading.")
        # Fallback to loading everything together
        results = torch.load(model_path)
        state_dict = results['model_state_dict']
        config = results['config']
        X_scaler = results['X_scaler']
        y_scaler = results['y_scaler']
    
    # Load data
    df = pd.read_csv(config['data_path'])
    df = df[df['genotype'] == config['genotype']]
    
    # Get trial indices
    trial_size = 1400
    num_trials = len(df) // trial_size
    
    # Create trial indices and shuffle with same seed
    trial_indices = np.arange(num_trials)
    np.random.seed(42)  # Same seed as in training
    np.random.shuffle(trial_indices)
    
    # Get test indices (last 15% of trials)
    train_size = int(0.7 * num_trials)
    val_size = int(0.15 * num_trials)
    test_indices = trial_indices[train_size + val_size:]
    
    if trial_id is not None:
        if trial_id not in test_indices:
            raise ValueError(f"Trial {trial_id} is not a test trial. Available test trials: {test_indices}")
        test_indices = [trial_id]
    
    # Get angle names for the specified leg
    angle_names = get_leg_angles(leg if leg else 'R-F')  # Default to R-F instead of R1
    
    # Calculate moving averages for velocities within each trial
    base_velocities = ['x_vel', 'y_vel', 'z_vel']
    for window in [5, 10, 20]:
        for vel in base_velocities:
            ma_values = []
            for trial in range(num_trials):
                start_idx = trial * trial_size
                end_idx = (trial + 1) * trial_size
                trial_data = df[vel].iloc[start_idx:end_idx]
                ma = trial_data.rolling(window=window, center=True, min_periods=1).mean()
                ma_values.extend(ma.tolist())
            df[f'{vel}_ma{window}'] = ma_values
    
    # Define input features
    velocity_features = [
        'x_vel_ma5', 'y_vel_ma5', 'z_vel_ma5',
        'x_vel_ma10', 'y_vel_ma10', 'z_vel_ma10',
        'x_vel_ma20', 'y_vel_ma20', 'z_vel_ma20',
        'x_vel', 'y_vel', 'z_vel'
    ]
    input_features = velocity_features + [config['input_features'][-1]]
    
    # Create model
    model = TemporalConvNet(
        input_size=len(input_features),
        output_size=len(angle_names),
        num_channels=config['num_channels'],
        kernel_size=config['kernel_size'],
        dropout=config['dropout'],
        sequence_length=config['sequence_length']
    )
    
    # Load model weights
    model.load_state_dict(state_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_angles = {}
    all_actual_angles = {}  # dictionary for actual angles
    
    for trial_id in test_indices:
        # Get trial data
        start_idx = trial_id * trial_size
        end_idx = (trial_id + 1) * trial_size
        trial_df = df.iloc[start_idx:end_idx]
        
        # Get features
        X = trial_df[input_features].values
        X_scaled = X_scaler.transform(X)
        
        predictions = []
        # Generate predictions for frames 400-1000
        for frame in range(400, 1001):
            start_frame = frame - config['sequence_length']
            window = X_scaled[start_frame:frame]
            if len(window) < config['sequence_length']:
                continue
            window_tensor = torch.FloatTensor(window.T).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(window_tensor)
                pred = pred.cpu().numpy().reshape(1, -1)
                pred = y_scaler.inverse_transform(pred)
                predictions.append(pred[0])
        
        all_angles[trial_id] = np.array(predictions)
        # Also extract the actual angles from the dataframe for the same frames
        actual = trial_df[angle_names].iloc[400:1001].values
        all_actual_angles[trial_id] = actual
        print(f"Generated predictions for trial {trial_id}")
    
    return all_angles, all_actual_angles

def visualize_combined_leg_movement(pred_angles, actual_angles, save_path=None, interval=200):
    """Create a 3D animation of all legs moving together."""
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set tighter axis limits for better zoom
    ax.set_xlim([-4, 4])  # Wider X limits to accommodate spread-out legs
    ax.set_ylim([-3, 3])  # Y limits for front-to-back spacing
    ax.set_zlim([2, -2])  # Z limits for height
    
    # Adjust the viewing angle for better perspective
    ax.view_init(elev=20, azim=45)
    
    # Add labels for orientation aligned with leg positions
    ax.text(3.5, 0, 0, 'LEFT', color='red', fontsize=14, fontweight='bold')
    ax.text(-3.5, 0, 0, 'RIGHT', color='red', fontsize=14, fontweight='bold')
    ax.text(0, 2.0, 0, 'FRONT', color='red', fontsize=14)    # Aligned with front legs
    ax.text(0, 0.0, 0, 'MIDDLE', color='red', fontsize=14)   # Aligned with middle legs
    ax.text(0, -2.0, 0, 'HIND', color='red', fontsize=14)    # Aligned with hind legs
    
    # Set title and grid
    ax.set_title('Leg Movements\nBlue: Actual, Red: Predicted', pad=20, fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Initial positions for predicted and actual movements
    pos_pred = np.array(create_leg_segments(pred_angles[0], 'R1'))
    pos_actual = np.array(create_leg_segments(actual_angles[0], 'R1'))
    
    # Plot predicted movement (line and joint points)
    line_pred, = ax.plot(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2], 'b-', lw=2)
    points_pred, = ax.plot(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2], 'ro')
    ax.set_title("Predicted Movement", fontsize=16)
    
    # Plot actual movement (line and joint points)
    line_actual, = ax.plot(pos_actual[:, 0], pos_actual[:, 1], pos_actual[:, 2], 'b-', lw=2)
    points_actual, = ax.plot(pos_actual[:, 0], pos_actual[:, 1], pos_actual[:, 2], 'ro')
    ax.set_title("Actual Movement", fontsize=16)
    
    # Set equal aspect ratio using union of positions for both subplots
    all_pos = np.concatenate([pos_pred, pos_actual], axis=0)
    max_range = (all_pos.max(axis=0) - all_pos.min(axis=0)).max() / 2.0
    mid = all_pos.mean(axis=0)
    for ax in [ax]:
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    def update_combined(frame):
        # Predicted movement update
        pos_p = np.array(create_leg_segments(pred_angles[frame], 'R1'))
        line_pred.set_data(pos_p[:, 0], pos_p[:, 1])
        line_pred.set_3d_properties(pos_p[:, 2])
        points_pred.set_data(pos_p[:, 0], pos_p[:, 1])
        points_pred.set_3d_properties(pos_p[:, 2])
        
        # Actual movement update
        pos_a = np.array(create_leg_segments(actual_angles[frame], 'R1'))
        line_actual.set_data(pos_a[:, 0], pos_a[:, 1])
        line_actual.set_3d_properties(pos_a[:, 2])
        points_actual.set_data(pos_a[:, 0], pos_a[:, 1])
        points_actual.set_3d_properties(pos_a[:, 2])
        
        return [line_pred, points_pred, line_actual, points_actual]
    
    anim = animation.FuncAnimation(
        fig, update_combined, frames=len(pred_angles), interval=interval, blit=True
    )
    
    if save_path:
        try:
            writer = animation.PillowWriter(fps=40)
            anim.save(save_path, writer=writer)
        except Exception as e:
            print(f"Failed to save with PillowWriter: {e}")
            try:
                anim.save(save_path, writer='imagemagick', fps=40)
            except Exception as e:
                print(f"Failed to save with imagemagick: {e}")
                anim.save(save_path, writer='pillow')
        plt.close()
    else:
        plt.show()

def visualize_all_legs(model_paths, trial_id=None, interval=50):
    """
    Visualize all 6 legs simultaneously
    model_paths: dictionary mapping leg names to their model paths
    trial_id: specific trial to visualize (optional)
    interval: animation interval in ms
    """
    # Create figure with 2x3 subplots for 6 legs
    fig = plt.figure(figsize=(20, 15))
    
    # Define leg order and positions in the subplot grid
    leg_positions = {
        'R-F': (0, 0), 'L-F': (0, 1),
        'R-M': (1, 0), 'L-M': (1, 1),
        'R-H': (2, 0), 'L-H': (2, 1)
    }
    
    # Store axes and initial positions for each leg
    axes = {}
    initial_positions = {}
    leg_data = {}
    
    # Load data for each leg
    for leg_name, model_path in model_paths.items():
        row, col = leg_positions[leg_name]
        ax = fig.add_subplot(3, 2, row*2 + col + 1, projection='3d')
        axes[leg_name] = ax
        
        # Load predictions and actual angles
        pred_angles_dict, actual_angles_dict = load_predicted_angles(
            model_path, trial_id=trial_id, leg=leg_name
        )
        
        # Store data
        leg_data[leg_name] = {
            'pred_angles': pred_angles_dict[list(pred_angles_dict.keys())[0]],
            'actual_angles': actual_angles_dict[list(actual_angles_dict.keys())[0]]
        }
        
        # Get initial positions
        pos_pred = np.array(create_leg_segments(leg_data[leg_name]['pred_angles'][0], leg_name[:2]))
        pos_actual = np.array(create_leg_segments(leg_data[leg_name]['actual_angles'][0], leg_name[:2]))
        
        # Plot predicted movement in red
        line_pred, = ax.plot(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2], 'r-', lw=2, label='Predicted')
        points_pred, = ax.plot(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2], 'ro')
        
        # Plot actual movement in blue
        line_actual, = ax.plot(pos_actual[:, 0], pos_actual[:, 1], pos_actual[:, 2], 'b-', lw=2, label='True')
        points_actual, = ax.plot(pos_actual[:, 0], pos_actual[:, 1], pos_actual[:, 2], 'bo')
        
        # Store lines for animation
        initial_positions[leg_name] = {
            'pred_lines': (line_pred, points_pred),
            'actual_lines': (line_actual, points_actual)
        }
        
        # Set title and labels
        ax.set_title(f"{leg_name} Leg")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        
        # Set equal aspect ratio
        max_range = np.array([
            pos_pred[:, 0].max()-pos_pred[:, 0].min(),
            pos_pred[:, 1].max()-pos_pred[:, 1].min(),
            pos_pred[:, 2].max()-pos_pred[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (pos_pred[:, 0].max()+pos_pred[:, 0].min()) * 0.5
        mid_y = (pos_pred[:, 1].max()+pos_pred[:, 1].min()) * 0.5
        mid_z = (pos_pred[:, 2].max()+pos_pred[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    def update(frame):
        lines = []
        for leg_name in leg_data:
            # Get positions for this frame
            pos_pred = np.array(create_leg_segments(leg_data[leg_name]['pred_angles'][frame], leg_name[:2]))
            pos_actual = np.array(create_leg_segments(leg_data[leg_name]['actual_angles'][frame], leg_name[:2]))
            
            # Update predicted lines (red)
            line_pred, points_pred = initial_positions[leg_name]['pred_lines']
            line_pred.set_data(pos_pred[:, 0], pos_pred[:, 1])
            line_pred.set_3d_properties(pos_pred[:, 2])
            points_pred.set_data(pos_pred[:, 0], pos_pred[:, 1])
            points_pred.set_3d_properties(pos_pred[:, 2])
            
            # Update actual lines (blue)
            line_actual, points_actual = initial_positions[leg_name]['actual_lines']
            line_actual.set_data(pos_actual[:, 0], pos_actual[:, 1])
            line_actual.set_3d_properties(pos_actual[:, 2])
            points_actual.set_data(pos_actual[:, 0], pos_actual[:, 1])
            points_actual.set_3d_properties(pos_actual[:, 2])
            
            lines.extend([line_pred, points_pred, line_actual, points_actual])
        
        return lines
    
    # Create animation
    n_frames = len(list(leg_data.values())[0]['pred_angles'])
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=interval, blit=True
    )
    
    return fig, anim

def main():
    """Main function to create leg movement visualization."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Visualize leg movements for specific trials')
    parser.add_argument('--genotype', type=str, help='Genotype to visualize (e.g., BPN, P9RT, P9LT)')
    parser.add_argument('--trial', type=int, help='Trial ID to visualize')
    parser.add_argument('--interval', type=int, default=100, help='Animation interval in ms (default: 100)')
    args = parser.parse_args()

    # Define base directories
    tcn_results_dir = Path('tcn_results')
    gifs_dir = Path('gifs')
    gifs_dir.mkdir(exist_ok=True, parents=True)
    
    # Find available model genotypes
    model_genotypes = [d.name for d in tcn_results_dir.iterdir() if d.is_dir()]
    if not model_genotypes:
        print("No model results found in tcn_results directory.")
        return

    # Handle genotype selection
    if args.genotype:
        if args.genotype not in model_genotypes:
            print(f"Error: Genotype '{args.genotype}' not found. Available genotypes:")
            for genotype in model_genotypes:
                print(f"  - {genotype}")
            return
        selected_genotype = args.genotype
    else:
        print("\nAvailable model genotypes:")
        for i, genotype in enumerate(model_genotypes):
            print(f"{i}: {genotype}")
        try:
            genotype_idx = int(input("\nSelect model genotype to visualize (enter number): "))
            if genotype_idx < 0 or genotype_idx >= len(model_genotypes):
                print("Invalid genotype index")
                return
            selected_genotype = model_genotypes[genotype_idx]
        except ValueError:
            print("Please enter a valid number")
            return

    genotype_dir = tcn_results_dir / selected_genotype
    
    # Define leg configurations
    leg_configs = {
        'R-F': 'R1', 'L-F': 'L1',
        'R-M': 'R2', 'L-M': 'L2',
        'R-H': 'R3', 'L-H': 'L3'
    }
    
    # Load data for all legs
    print(f"\nLoading leg data for {selected_genotype}...")
    actual_angles = {}
    pred_angles = {}
    
    # First, find available trials from any leg
    first_leg_path = None
    for leg_name in leg_configs:
        model_path = genotype_dir / f"{leg_name}_leg/best_model.pt"
        if model_path.exists():
            first_leg_path = model_path
            break
    
    if not first_leg_path:
        print("No model files found.")
        return
    
    # Load first model to get trial information
    try:
        checkpoint = torch.load(first_leg_path)
        config = checkpoint['config']
        
        # Load data
        df = pd.read_csv(config['data_path'])
        df = df[df['genotype'] == config['genotype']]
        
        # Get trial indices
        trial_size = 1400
        num_trials = len(df) // trial_size
        
        # Create trial indices and shuffle with same seed
        trial_indices = np.arange(num_trials)
        np.random.seed(42)  # Same seed as in training
        np.random.shuffle(trial_indices)
        
        # Get test indices (last 15% of trials)
        train_size = int(0.7 * num_trials)
        val_size = int(0.15 * num_trials)
        test_indices = trial_indices[train_size + val_size:]
        
        print(f"\nFound {len(test_indices)} test trials")
        
        # Handle trial selection
        if args.trial is not None:
            if args.trial not in test_indices:
                print(f"Error: Trial {args.trial} is not a test trial.")
                print("Available test trials:", sorted(test_indices))
                return
            selected_trial = args.trial
        else:
            print("Available test trials:", sorted(test_indices))
            trial_idx = int(input(f"Select trial to visualize (0-{len(test_indices)-1}): "))
            if trial_idx < 0 or trial_idx >= len(test_indices):
                print("Invalid trial index")
                return
            selected_trial = test_indices[trial_idx]
            
    except Exception as e:
        print(f"Error loading model data: {e}")
        return
    
    print(f"\nProcessing trial {selected_trial} for {selected_genotype}...")
    
    # Load data and generate predictions for each leg
    for leg_name, leg_prefix in leg_configs.items():
        model_path = genotype_dir / f"{leg_name}_leg/best_model.pt"
        if not model_path.exists():
            print(f"Skipping {leg_name} - model not found")
            continue
        
        try:
            pred_angles_dict, actual_angles_dict = load_predicted_angles(
                model_path, trial_id=selected_trial, leg=leg_name
            )
            
            # Get the angles for this trial
            trial_pred = pred_angles_dict[selected_trial]
            trial_actual = actual_angles_dict[selected_trial]
            
            # Store predictions and actual values
            for i, angle_name in enumerate(get_leg_angles(leg_name)):
                actual_angles[angle_name] = trial_actual[:, i]
                pred_angles[angle_name] = trial_pred[:, i]
            
            print(f"Generated predictions for {leg_name}")
            
        except Exception as e:
            print(f"Error processing {leg_name}: {e}")
            continue
    
    if not actual_angles or not pred_angles:
        print("No predictions generated. Cannot create visualization.")
        return
    
    # Create animation
    save_path = gifs_dir / f'leg_movement_trial_{selected_trial}_{selected_genotype}.gif'
    print("\nCreating animation...")
    visualize_leg_movement(actual_angles, pred_angles, save_path=save_path, interval=args.interval)
    print(f"\nAnimation saved to: {save_path}")

if __name__ == "__main__":
    main() 