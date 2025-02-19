# correlation.py

import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from utils.data import filter_frames

def calculate_enhanced_features(df):
    """Calculate enhanced features including lagged velocities and integrated positions."""
    # Initialize features dictionary to avoid DataFrame fragmentation
    features_dict = {}
    
    # Original velocities
    features_dict['x_vel'] = df['x_vel'].values
    features_dict['y_vel'] = df['y_vel'].values
    features_dict['z_vel'] = df['z_vel'].values
    
    # Calculate positions by integrating velocities
    dt = 1/200  # 200Hz sampling rate
    trial_size = 600  # After filtering (1000 - 400 = 600)
    num_trials = len(df) // trial_size
    print(f"\nCalculating positions for {num_trials} trials")
    
    # Initialize position arrays
    x_pos = np.zeros_like(df['x_vel'].values)
    y_pos = np.zeros_like(df['y_vel'].values)
    z_pos = np.zeros_like(df['z_vel'].values)
    
    # Calculate positions for each trial separately to avoid accumulating error
    for trial in range(num_trials):
        start_idx = trial * trial_size
        end_idx = (trial + 1) * trial_size
        
        # Get velocities for this trial
        x_vel_trial = df['x_vel'].values[start_idx:end_idx]
        y_vel_trial = df['y_vel'].values[start_idx:end_idx]
        z_vel_trial = df['z_vel'].values[start_idx:end_idx]
        
        # Integrate velocities to get positions (cumulative trapezoidal integration)
        x_pos[start_idx:end_idx] = np.cumsum(x_vel_trial) * dt
        y_pos[start_idx:end_idx] = np.cumsum(y_vel_trial) * dt
        z_pos[start_idx:end_idx] = np.cumsum(z_vel_trial) * dt
        
        # Remove mean to center positions around zero for each trial
        x_pos[start_idx:end_idx] -= np.mean(x_pos[start_idx:end_idx])
        y_pos[start_idx:end_idx] -= np.mean(y_pos[start_idx:end_idx])
        z_pos[start_idx:end_idx] -= np.mean(z_pos[start_idx:end_idx])
    
    # Add positions to features
    features_dict['x_pos'] = x_pos
    features_dict['y_pos'] = y_pos
    features_dict['z_pos'] = z_pos
    
    # Add lagged velocities (both positive and negative lags)
    lag_values = [1, 2, 3, 5, 10, 20]  # Different lag amounts
    
    # Pre-compute all velocity and position arrays
    vel_arrays = {
        'x': df['x_vel'].values,
        'y': df['y_vel'].values,
        'z': df['z_vel'].values
    }
    pos_arrays = {
        'x': x_pos,
        'y': y_pos,
        'z': z_pos
    }
    
    for lag in lag_values:
        for coord in ['x', 'y', 'z']:
            # Forward lags (future values)
            vel_arr = np.roll(vel_arrays[coord], -lag)
            pos_arr = np.roll(pos_arrays[coord], -lag)
            features_dict[f'{coord}_vel_lag_plus_{lag}'] = vel_arr
            features_dict[f'{coord}_pos_lag_plus_{lag}'] = pos_arr
            
            # Backward lags (past values)
            vel_arr = np.roll(vel_arrays[coord], lag)
            pos_arr = np.roll(pos_arrays[coord], lag)
            features_dict[f'{coord}_vel_lag_minus_{lag}'] = vel_arr
            features_dict[f'{coord}_pos_lag_minus_{lag}'] = pos_arr
    
    # Calculate moving averages
    windows = [5, 10, 20]
    for window in windows:
        for coord in ['x', 'y', 'z']:
            # Velocity moving averages
            ma_vel = pd.Series(vel_arrays[coord]).rolling(window=window, center=True).mean().values
            features_dict[f'{coord}_vel_ma{window}'] = ma_vel
            
            # Position moving averages
            ma_pos = pd.Series(pos_arrays[coord]).rolling(window=window, center=True).mean().values
            features_dict[f'{coord}_pos_ma{window}'] = ma_pos
    
    # Calculate derived velocities and positions
    features_dict['velocity_magnitude'] = np.sqrt(
        vel_arrays['x']**2 + vel_arrays['y']**2 + vel_arrays['z']**2
    )
    features_dict['position_magnitude'] = np.sqrt(
        pos_arrays['x']**2 + pos_arrays['y']**2 + pos_arrays['z']**2
    )
    
    features_dict['xy_velocity'] = np.sqrt(vel_arrays['x']**2 + vel_arrays['y']**2)
    features_dict['xy_position'] = np.sqrt(pos_arrays['x']**2 + pos_arrays['y']**2)
    
    features_dict['xz_velocity'] = np.sqrt(vel_arrays['x']**2 + vel_arrays['z']**2)
    features_dict['xz_position'] = np.sqrt(pos_arrays['x']**2 + pos_arrays['y']**2)
    
    # Calculate accelerations using central difference
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

def load_data(file_path, genotype=None):
    """Load and preprocess the dataset from a CSV file."""
    try:
        # Load the full data
        df = pd.read_csv(file_path)
        print(f"Raw data loaded successfully with shape: {df.shape}")
        
        # Filter for genotype
        if genotype:
            df = df[df['genotype'] == genotype]
            print(f"Filtered data for {genotype} genotype, new shape: {df.shape}")
        
        # Calculate velocities for TaG points
        tag_cols = [col for col in df.columns if 'TaG' in col]
        vel = df[tag_cols].astype(float)  # Ensure numeric type
        
        velocities = vel.diff().fillna(0) / vel.index.to_series().diff().fillna(1).values[:, None]
        velocities.columns = [col.replace('TaG', 'Vel') for col in velocities.columns]
        
        # Add velocities to the dataframe
        df = pd.concat([df, velocities], axis=1)
        print(f"Added velocity features, new shape: {df.shape}")
        
        # Filter frames 400-1000 from each trial
        trial_size = 1400
        num_trials = len(df) // trial_size
        print(f"Number of trials detected: {num_trials}")
        
        filtered_rows = []
        for trial in range(num_trials):
            start_idx = trial * trial_size + 400
            end_idx = trial * trial_size + 1000
            trial_data = df.iloc[start_idx:end_idx].copy()  # Create a copy to avoid fragmentation
            filtered_rows.append(trial_data)
        
        df = pd.concat(filtered_rows, axis=0, ignore_index=True)
        print(f"Filtered data to frames 400-1000, new shape: {df.shape}")
        
        # Convert joint angles to numeric type
        joint_cols = [col for col in df.columns if col.endswith(('_flex', '_rot', '_abduct'))]
        df[joint_cols] = df[joint_cols].astype(float)
        
        # Verify data integrity
        print("\nVerifying data integrity:")
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        print(f"Any NaN values in joint angles: {df[joint_cols].isna().any().any()}")
        print(f"Any NaN values in TaG points: {df[tag_cols].isna().any().any()}")
        
        return df
        
    except Exception as e:
        print(f"Error loading/preprocessing data: {e}")
        raise

def compute_correlation(df):
    """Compute the Pearson correlation matrix between inputs and outputs."""
    # Define output features (joint angles)
    output_features = [col for col in df.columns 
                      if col.endswith(('_flex', '_rot', '_abduct'))]
    
    # Calculate enhanced features
    print("\nCalculating enhanced features (positions, velocities, etc.)...")
    enhanced_features = calculate_enhanced_features(df)
    
    # Combine original data with enhanced features
    df_combined = pd.concat([df, enhanced_features], axis=1)
    
    # Get all input features: TaG points, velocities, positions, accelerations, etc.
    # Make sure to only get unique columns
    all_features = [col for col in df_combined.select_dtypes(include=[np.number]).columns 
                   if any(x in col.lower() for x in ['tag', 'vel', 'pos', 'acc', 'jerk', 'magnitude'])]
    input_features = list(dict.fromkeys([col for col in all_features 
                                       if col not in output_features 
                                       and col != 'genotype']))
    
    print(f"Number of input features: {len(input_features)}")
    print(f"Number of joint angles: {len(output_features)}")
    print("\nInput features:")
    for feat in input_features:
        print(f"  {feat}")
    print("\nJoint angles:")
    for feat in output_features:
        print(f"  {feat}")
    
    # Print shape information for debugging
    print(f"\nShape of df_combined: {df_combined.shape}")
    print(f"Length of first input feature '{input_features[0]}': {len(df_combined[input_features[0]])}")
    print(f"Length of first output feature '{output_features[0]}': {len(df_combined[output_features[0]])}")
    
    # Initialize correlation matrix
    correlation_matrix = pd.DataFrame(
        np.zeros((len(input_features), len(output_features))),
        index=input_features,
        columns=output_features
    )
    
    # Calculate correlations between each input-output pair
    for input_feat in input_features:
        for output_feat in output_features:
            try:
                # Get the data as numpy arrays, ensuring we get 1D arrays
                x = df_combined[input_feat].to_numpy().ravel()
                y = df_combined[output_feat].to_numpy().ravel()
                
                # Ensure both arrays have the same length
                min_len = min(len(x), len(y))
                x = x[:min_len]
                y = y[:min_len]
                
                # Remove any NaN values
                mask = ~(np.isnan(x) | np.isnan(y))
                x = x[mask]
                y = y[mask]
                
                # Calculate Pearson correlation using numpy corrcoef
                if len(x) > 0 and len(y) > 0:
                    correlation = np.corrcoef(x, y)[0, 1]  # Changed from spearmanr to corrcoef
                else:
                    correlation = np.nan
                correlation_matrix.loc[input_feat, output_feat] = correlation
                
            except Exception as e:
                print(f"\nError calculating correlation between {input_feat} and {output_feat}: {e}")
                print(f"Shape of {input_feat}: {df_combined[input_feat].shape}")
                print(f"Shape of {output_feat}: {df_combined[output_feat].shape}")
                correlation_matrix.loc[input_feat, output_feat] = np.nan
    
    return correlation_matrix

def plot_correlation_heatmap(correlation_matrix, save_path=None):
    """Plot and save Pearson correlation heatmap between inputs and outputs."""
    # Create output directory if it doesn't exist
    if save_path:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with size proportional to the number of features
    plt.figure(figsize=(max(24, len(correlation_matrix.columns)), 
                       max(12, len(correlation_matrix.index) * 0.3)))
    
    # Plot heatmap
    sns.heatmap(correlation_matrix, 
                annot=True, 
                fmt=".2f", 
                cmap='coolwarm', 
                center=0,
                vmin=-1, 
                vmax=1,
                cbar_kws={"shrink": .8})
    
    plt.title('TaG Points and Velocities vs Joint Angles Pearson Correlation')
    plt.xlabel('Joint Angles')
    plt.ylabel('TaG Points and Their Velocities')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Rotate y-axis labels for better readability if there are many features
    if len(correlation_matrix.index) > 30:
        plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def analyze_correlations(correlation_matrix, threshold=0.3, output_dir=None):
    """Analyze and print significant Pearson correlations between inputs and outputs."""
    print("\nSignificant Pearson Correlations Analysis:")
    print("-" * 50)
    
    # Store significant correlations
    significant_correlations = []
    
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
                significant_correlations.append({
                    'Feature': input_feature,
                    'Joint Angle': joint_angle,
                    'Correlation': corr
                })
    
    # Sort significant correlations by absolute correlation value
    significant_correlations.sort(key=lambda x: abs(x['Correlation']), reverse=True)
    
    # Save significant correlations to CSV if output directory is provided
    if output_dir and significant_correlations:
        sig_corr_df = pd.DataFrame(significant_correlations)
        sig_corr_path = output_dir / 'significant_correlations.csv'
        sig_corr_df.to_csv(sig_corr_path, index=False)
        print(f"\nSaved significant correlations to: {sig_corr_path}")
    
    # Print top correlations
    print("\nTop 20 strongest input-output correlations (Pearson):")
    for i, corr in enumerate(significant_correlations[:20]):
        print(f"{corr['Feature']} -- {corr['Joint Angle']}: {corr['Correlation']:.3f}")
    
    return significant_correlations

def get_available_data_path():
    """Try multiple possible data paths and return the first available one."""
    possible_paths = [
        "Z:/Divya/TEMP_transfers/toAni/BPN_P9LT_P9RT_flyCoords.csv",  # Network drive path
        "/Users/anivenkat/Downloads/BPN_P9LT_P9RT_flyCoords.csv",      # Local Mac path
        "C:/Users/bidayelab/Downloads/BPN_P9LT_P9RT_flyCoords.csv"     # Local Windows path
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            print(f"Using data file: {path}")
            return path
    
    raise FileNotFoundError("Could not find the data file in any of the expected locations")

def main():
    try:
        # Get the first available data path
        file_path = get_available_data_path()
        
        parent_dir = Path("correlation_results")
        parent_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each genotype
        genotypes = ['P9RT', 'BPN', 'P9LT']
        
        for genotype in genotypes:
            print(f"\nProcessing {genotype} genotype...")
            output_dir = parent_dir / genotype
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load and process data for this genotype
            df = load_data(file_path, genotype)
            if df is not None:
                # Compute Pearson correlations between inputs and outputs
                correlation_matrix = compute_correlation(df)
                
                # Plot heatmap
                plot_correlation_heatmap(
                    correlation_matrix,
                    save_path=str(output_dir / f"pearson_input_output_correlation_{genotype}.png")
                )
                
                # Analyze correlations and save to CSV
                print(f"\nPearson Correlation Analysis for {genotype}:")
                analyze_correlations(correlation_matrix, output_dir=output_dir)
                
                # Save correlation matrix to CSV
                correlation_matrix.to_csv(output_dir / f"pearson_correlation_matrix_{genotype}.csv")
                print(f"\nResults saved to {output_dir}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data file is available in one of the expected locations.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    main()