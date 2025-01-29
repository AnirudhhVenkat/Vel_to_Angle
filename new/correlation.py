# correlation.py

import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from utils.data import filter_frames

def calculate_enhanced_features(df):
    """Calculate enhanced features including lagged velocities."""
    features = pd.DataFrame()
    
    # Original velocities
    features['x_vel'] = df['x_vel']
    features['y_vel'] = df['y_vel']
    features['z_vel'] = df['z_vel']
    
    # Add lagged velocities (both positive and negative lags)
    lag_values = [1, 2, 3, 5, 10, 20]  # Different lag amounts
    
    for lag in lag_values:
        # Forward lags (future values)
        features[f'x_vel_lag_plus_{lag}'] = df['x_vel'].shift(-lag)
        features[f'y_vel_lag_plus_{lag}'] = df['y_vel'].shift(-lag)
        features[f'z_vel_lag_plus_{lag}'] = df['z_vel'].shift(-lag)
        
        # Backward lags (past values)
        features[f'x_vel_lag_minus_{lag}'] = df['x_vel'].shift(lag)
        features[f'y_vel_lag_minus_{lag}'] = df['y_vel'].shift(lag)
        features[f'z_vel_lag_minus_{lag}'] = df['z_vel'].shift(lag)
    
    # Calculate moving averages
    windows = [5, 10, 20]
    for window in windows:
        features[f'x_vel_ma{window}'] = df['x_vel'].rolling(window=window, center=True).mean()
        features[f'y_vel_ma{window}'] = df['y_vel'].rolling(window=window, center=True).mean()
        features[f'z_vel_ma{window}'] = df['z_vel'].rolling(window=window, center=True).mean()
    
    # Calculate derived velocities
    features['velocity_magnitude'] = np.sqrt(
        df['x_vel']**2 + df['y_vel']**2 + df['z_vel']**2
    )
    features['xy_velocity'] = np.sqrt(
        df['x_vel']**2 + df['y_vel']**2
    )
    features['xz_velocity'] = np.sqrt(
        df['x_vel']**2 + df['z_vel']**2
    )
    
    # Calculate accelerations using central difference
    dt = 1/200  # 200Hz sampling rate
    
    # Initialize acceleration arrays with zeros
    accelerations = np.zeros_like(df['x_vel'])
    
    # Calculate accelerations using central difference (more accurate than forward difference)
    accelerations[1:-1] = (df['x_vel'].values[2:] - df['x_vel'].values[:-2]) / (2 * dt)
    features['x_acc'] = accelerations
    
    accelerations[1:-1] = (df['y_vel'].values[2:] - df['y_vel'].values[:-2]) / (2 * dt)
    features['y_acc'] = accelerations
    
    accelerations[1:-1] = (df['z_vel'].values[2:] - df['z_vel'].values[:-2]) / (2 * dt)
    features['z_acc'] = accelerations
    
    # Calculate total acceleration magnitude
    features['acceleration_magnitude'] = np.sqrt(
        features['x_acc']**2 + features['y_acc']**2 + features['z_acc']**2
    )
    
    # Calculate jerk (derivative of acceleration)
    jerk = np.zeros_like(accelerations)
    jerk[1:-1] = (features['acceleration_magnitude'].values[2:] - 
                  features['acceleration_magnitude'].values[:-2]) / (2 * dt)
    features['jerk_magnitude'] = jerk
    
    # Handle NaN values at trial boundaries
    # First, identify trial boundaries
    trial_size = 600  # After filtering (1000 - 400 = 600)
    num_trials = len(df) // trial_size
    
    for trial in range(num_trials):
        start_idx = trial * trial_size
        end_idx = (trial + 1) * trial_size
        
        # Forward fill within each trial
        features.iloc[start_idx:end_idx] = features.iloc[start_idx:end_idx].ffill()
        # Backward fill within each trial
        features.iloc[start_idx:end_idx] = features.iloc[start_idx:end_idx].bfill()
    
    return features

def load_data(file_path, genotype=None):
    """Load and preprocess the dataset from a CSV file efficiently."""
    try:
        # Load the full data first to get all columns
        df = pd.read_csv(file_path)
        print(f"Raw data loaded successfully with shape: {df.shape}")
        
        if genotype:
            df = df[df['genotype'] == genotype]
            print(f"Filtered data for {genotype} genotype, new shape: {df.shape}")    
        
        # Filter frames 400-1000 from each trial
        trial_size = 1400
        num_trials = len(df) // trial_size
        
        filtered_rows = []
        for trial in range(num_trials):
            start_idx = trial * trial_size + 400
            end_idx = trial * trial_size + 1000
            filtered_rows.append(df.iloc[start_idx:end_idx])
        
        df = pd.concat(filtered_rows, axis=0, ignore_index=True)
        print(f"Filtered data to frames 400-1000, new shape: {df.shape}")
        
        # Calculate enhanced features
        enhanced_features = calculate_enhanced_features(df)
        
        # Merge enhanced features with joint angles
        joint_cols = [col for col in df.columns if col.endswith(('_flex', '_rot', '_abduct'))]
        
        # Create final dataframe with both features and joint angles
        final_df = pd.concat([enhanced_features, df[joint_cols]], axis=1)
        
        # Calculate z-scores efficiently
        feature_cols = [col for col in enhanced_features.columns]
        final_df[feature_cols] = final_df[feature_cols].apply(stats.zscore)
        final_df[joint_cols] = final_df[joint_cols].apply(stats.zscore)
        
        print(f"Final preprocessed data shape: {final_df.shape}")
        return final_df
        
    except Exception as e:
        print(f"Error loading/preprocessing data: {e}")
        raise  # Re-raise the exception for better error tracking

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

def analyze_correlations(correlation_matrix, threshold=0.3, output_dir=None):
    """Analyze and print significant correlations between inputs and outputs."""
    print("\nSignificant Correlations Analysis:")
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
    print("\nTop 20 strongest input-output correlations:")
    for i, corr in enumerate(significant_correlations[:20]):
        print(f"{corr['Feature']} -- {corr['Joint Angle']}: {corr['Correlation']:.3f}")
    
    return significant_correlations

def main():
    # Specify paths
    file_path = "/Users/anivenkat/Downloads/BPN_P9LT_P9RT_flyCoords.csv"
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
            # Compute correlations between inputs and outputs
            correlation_matrix = compute_correlation(df)
            
            # Plot heatmap
            plot_correlation_heatmap(
                correlation_matrix,
                save_path=str(output_dir / f"enhanced_input_output_correlation_{genotype}.png")
            )
            
            # Analyze correlations and save to CSV
            print(f"\nCorrelation Analysis for {genotype}:")
            analyze_correlations(correlation_matrix, output_dir=output_dir)
            
            # Save correlation matrix to CSV
            correlation_matrix.to_csv(output_dir / f"correlation_matrix_{genotype}.csv")
            print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()