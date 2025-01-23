# correlation.py

import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from utils.data import filter_frames

def calculate_enhanced_features(df):
    """Calculate enhanced motion features from velocity data."""
    # Original velocities
    velocity_cols = ['x_vel', 'y_vel', 'z_vel']
    velocities = df[velocity_cols].values
    
    # 1. Total velocity magnitude
    velocity_magnitude = np.sqrt(np.sum(velocities**2, axis=1))
    df['velocity_magnitude'] = velocity_magnitude
    
    # 2. Accelerations
    accelerations = np.diff(velocities, axis=0, prepend=velocities[0:1])
    acc_cols = ['x_acc', 'y_acc', 'z_acc']
    for col, acc_array in zip(acc_cols, accelerations.T):
        df[col] = acc_array
    
    # 3. Jerk (rate of change of acceleration)
    jerk = np.diff(accelerations, axis=0, prepend=accelerations[0:1])
    jerk_cols = ['x_jerk', 'y_jerk', 'z_jerk']
    for col, jerk_array in zip(jerk_cols, jerk.T):
        df[col] = jerk_array
    
    # 4. Moving averages for trend detection
    window_sizes = [5, 10, 20]
    for w in window_sizes:
        for i, vel_col in enumerate(velocity_cols):
            ma = pd.Series(velocities[:, i]).rolling(window=w, min_periods=1).mean().values
            df[f'{vel_col}_ma{w}'] = ma
    
    # 5. Planar velocities
    df['xy_velocity'] = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
    df['yz_velocity'] = np.sqrt(velocities[:, 1]**2 + velocities[:, 2]**2)
    df['xz_velocity'] = np.sqrt(velocities[:, 0]**2 + velocities[:, 2]**2)
    
    # 6. Angular velocities
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
    xy_angular_vel = pd.Series(xy_angular_vel).rolling(window=window_size, min_periods=1, center=True).mean().values
    xz_angular_vel = pd.Series(xz_angular_vel).rolling(window=window_size, min_periods=1, center=True).mean().values
    yz_angular_vel = pd.Series(yz_angular_vel).rolling(window=window_size, min_periods=1, center=True).mean().values
    
    df['xy_angular_vel'] = xy_angular_vel
    df['xz_angular_vel'] = xz_angular_vel
    df['yz_angular_vel'] = yz_angular_vel
    
    # 7. Direction changes
    velocity_directions = np.sum(velocities[1:] * velocities[:-1], axis=1)
    velocity_directions = np.pad(velocity_directions, (1,0), mode='edge')
    df['velocity_direction_change'] = velocity_directions
    
    # 8. Phase information
    df['phase'] = (df['fnum'].values % 1400) / 1400
    
    return df

def load_data(file_path):
    """Load and preprocess the dataset from a CSV file efficiently."""
    try:
        # Load only the columns we need
        # First, read just one row to get column names
        df_sample = pd.read_csv(file_path, nrows=1)
        
        # Identify relevant columns
        velocity_cols = ['x_vel', 'y_vel', 'z_vel']
        joint_cols = [col for col in df_sample.columns 
                     if col.endswith(('_flex', '_rot', '_abduct'))]
        frame_cols = ['fnum']  # needed for filter_frames
        
        usecols = velocity_cols + joint_cols + frame_cols
        
        # Load data with optimizations
        df = pd.read_csv(
            file_path,
            usecols=usecols,
            dtype={col: np.float32 for col in usecols if col != 'fnum'},
            engine='python',  # Use python engine for better compatibility
        )
        print(f"Raw data loaded successfully with shape: {df.shape}")
        
        # Apply preprocessing steps
        df = filter_frames(df)
        if 'index' in df.columns:
            df = df.drop(columns=['index'])
        
        # Calculate enhanced features
        df = calculate_enhanced_features(df)
        
        # Get all input feature columns
        input_cols = [col for col in df.columns if col not in joint_cols and col != 'fnum']
        
        # Calculate z-scores efficiently
        df[joint_cols] = df[joint_cols].apply(stats.zscore)
        df[input_cols] = df[input_cols].apply(stats.zscore)
        
        # Drop the frame number column as it's no longer needed
        df = df.drop(columns=['fnum'])
        
        print(f"Preprocessed data shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading/preprocessing data: {e}")
        return None

def compute_correlation(df):
    """Compute the correlation matrix between inputs and outputs."""
    # Define input features (all columns except joint angles)
    output_features = [col for col in df.columns 
                      if col.endswith(('_flex', '_rot', '_abduct'))]
    input_features = [col for col in df.columns if col not in output_features]
    
    print(f"Number of input features: {len(input_features)}")
    print(f"Number of output features: {len(output_features)}")
    print("\nInput features:")
    for feat in input_features:
        print(f"  {feat}")
    print("\nOutput features:")
    for feat in output_features:
        print(f"  {feat}")
    
    # Compute correlation matrix directly
    correlation_matrix = pd.DataFrame(
        np.corrcoef(df[input_features].values.T, df[output_features].values.T)[:len(input_features), len(input_features):],
        index=input_features,
        columns=output_features
    )
    
    return correlation_matrix

def plot_correlation_heatmap(correlation_matrix, save_path=None):
    """Plot and save correlation heatmap between inputs and outputs."""
    # Create output directory if it doesn't exist
    if save_path:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with appropriate size
    plt.figure(figsize=(24, 12))  # Increased height for more features
    
    # Plot heatmap
    sns.heatmap(correlation_matrix, 
                annot=True, 
                fmt=".2f", 
                cmap='coolwarm', 
                center=0,
                vmin=-1, 
                vmax=1,
                cbar_kws={"shrink": .8})
    
    plt.title('Enhanced Input Features vs Joint Angles Correlation')
    plt.xlabel('Joint Angles (Outputs)')
    plt.ylabel('Enhanced Motion Features (Inputs)')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def analyze_correlations(correlation_matrix, threshold=0.3):
    """Analyze and print significant correlations between inputs and outputs."""
    print("\nSignificant Correlations Analysis:")
    print("-" * 50)
    
    # For each input feature
    for input_feature in correlation_matrix.index:
        strong_correlations = correlation_matrix.loc[input_feature][
            abs(correlation_matrix.loc[input_feature]) > threshold
        ]
        
        if not strong_correlations.empty:
            print(f"\nStrong correlations for {input_feature}:")
            # Sort by absolute correlation value
            strong_correlations = strong_correlations.reindex(
                strong_correlations.abs().sort_values(ascending=False).index
            )
            for joint_angle, corr in strong_correlations.items():
                print(f"  {joint_angle}: {corr:.3f}")
    
    # Find the top correlations overall
    print("\nTop 20 strongest input-output correlations:")  # Increased to top 20
    flat_corr = correlation_matrix.unstack()
    strongest_corr = flat_corr[abs(flat_corr).sort_values(ascending=False).index][:20]
    
    for (input_feat, output_feat), corr in strongest_corr.items():
        print(f"{input_feat} -- {output_feat}: {corr:.3f}")

def main():
    # Specify paths
    file_path = "Z:/Divya/TEMP_transfers/toAni/BPN_P9LT_P9RT_flyCoords.csv"
    output_dir = Path("correlation_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and process data
    df = load_data(file_path)
    if df is not None:
        # Compute correlations between inputs and outputs
        correlation_matrix = compute_correlation(df)
        
        # Plot heatmap
        plot_correlation_heatmap(
            correlation_matrix,
            save_path=str(output_dir / "enhanced_input_output_correlation.png")
        )
        
        # Analyze correlations
        analyze_correlations(correlation_matrix)
        
        # Save correlation matrix to CSV
        correlation_matrix.to_csv(output_dir / "enhanced_input_output_correlation.csv")
        print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()