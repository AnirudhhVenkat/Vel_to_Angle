from utils.data import prepare_data
import torch
import numpy as np

config = {
    'data_path': "Z:/Divya/TEMP_transfers/toAni/BPN_P9LT_P9RT_flyCoords.csv",
    'window_size': 200,
    'split_ratio': 0.8,
    'test_size': 7200,
    'batch_size': 64,
    'use_angular_velocities': False  # Set to True if you want the additional features
}

train_loader, val_loader, test_loader = prepare_data(config)

print("\nChecking data structure:")
print(f"Window size: {config['window_size']}")
print(f"Trial size: 600")  # Hardcoded in data.py

# Get a batch from train loader
inputs, targets = next(iter(train_loader))

print("\nTensor shapes:")
print(f"Input shape: {inputs.shape}")
print("  - Batch size: {inputs.shape[0]}")
print("  - Sequence/Window length: {inputs.shape[1]}")
print("  - Number of features: {inputs.shape[2]}")
print(f"Target shape: {targets.shape}")

# Verify window structure
print("\nVerifying window structure:")
print("First window velocities (showing first 3 timesteps of vx):")
for t in range(3):
    print(f"Timestep {t}: {inputs[0, t, 0].item():.4f}")  # vx is first feature

print("\nVerifying targets correspond to last timestep:")
print("Last timestep joint angles:")
for i, angle in enumerate(['L2B_rot', 'R3A_rot', 'R3A_flex', 'R1B_rot', 'R2B_rot', 'L3A_rot']):
    print(f"{angle}: {targets[0, i].item():.4f}")

# Print dataset sizes
print("\nDataset sizes:")
print(f"Train set: {len(train_loader.dataset)} windows")
print(f"Val set: {len(val_loader.dataset)} windows")
print(f"Test set: {len(test_loader.dataset)} windows")

# Verify feature counts
velocity_features = [
    'x_vel_ma5', 'y_vel_ma5', 'z_vel_ma5',     # MA5 velocities
    'x_vel_ma10', 'y_vel_ma10', 'z_vel_ma10',  # MA10 velocities
    'x_vel_ma20', 'y_vel_ma20', 'z_vel_ma20',  # MA20 velocities
    'velocity_magnitude',                       # Velocity magnitude
    'xy_velocity', 'xz_velocity',              # Planar velocities
    'x_vel', 'y_vel', 'z_vel'                  # Original velocities
]

print("\nFeature verification:")
print(f"Number of velocity features: {len(velocity_features)} (should match input features: {inputs.shape[2]})")
print(f"Number of joint angles: 6 (should match target features: {targets.shape[1]})")

# Verify data is normalized
print("\nChecking data normalization:")
print("Input statistics (should be roughly mean=0, std=1):")
means = inputs.mean(dim=(0,1)).numpy()
stds = inputs.std(dim=(0,1)).numpy()
print(f"Mean range: [{means.min():.3f}, {means.max():.3f}]")
print(f"Std range: [{stds.min():.3f}, {stds.max():.3f}]")

print("\nTarget statistics (should be roughly mean=0, std=1):")
means = targets.mean(dim=0).numpy()
stds = targets.std(dim=0).numpy()
print(f"Mean range: [{means.min():.3f}, {means.max():.3f}]")
print(f"Std range: [{stds.min():.3f}, {stds.max():.3f}]")

