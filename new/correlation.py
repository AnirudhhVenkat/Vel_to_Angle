# correlation.py

import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from utils.data import filter_frames

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
            engine='c',  # Use C engine for faster parsing
        )
        print(f"Raw data loaded successfully with shape: {df.shape}")
        
        # Apply preprocessing steps
        df = filter_frames(df)
        if 'index' in df.columns:
            df = df.drop(columns=['index'])
        
        # Calculate z-scores efficiently
        df[joint_cols] = df[joint_cols].apply(stats.zscore)
        df[velocity_cols] = df[velocity_cols].apply(stats.zscore)
        
        # Calculate accelerations efficiently using numpy
        acc_arrays = np.diff(df[velocity_cols].values, axis=0, prepend=0)
        
        # Add acceleration columns
        acc_cols = ['x_acc', 'y_acc', 'z_acc']
        for col, acc_array in zip(acc_cols, acc_arrays.T):
            df[col] = acc_array
        
        # Drop the frame number column as it's no longer needed
        df = df.drop(columns=['fnum'])
        
        print(f"Preprocessed data shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading/preprocessing data: {e}")
        return None

def compute_correlation(df):
    """Compute the correlation matrix between inputs and outputs."""
    # Define input features (velocities and accelerations)
    input_features = ['x_vel', 'y_vel', 'z_vel', 'x_acc', 'y_acc', 'z_acc']
    
    # Get joint angle columns (outputs) - only those ending with _flex, _rot, or _abduct
    output_features = [col for col in df.columns 
                      if col.endswith(('_flex', '_rot', '_abduct'))]
    
    print(f"Number of input features: {len(input_features)}")
    print(f"Number of output features: {len(output_features)}")
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
    plt.figure(figsize=(24, 8))
    
    # Plot heatmap
    sns.heatmap(correlation_matrix, 
                annot=True, 
                fmt=".2f", 
                cmap='coolwarm', 
                center=0,
                vmin=-1, 
                vmax=1,
                cbar_kws={"shrink": .8})
    
    plt.title('Input Features vs Joint Angles Correlation')
    plt.xlabel('Joint Angles (Outputs)')
    plt.ylabel('Velocities and Accelerations (Inputs)')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def analyze_correlations(correlation_matrix, threshold=0.3):  # Lowered threshold to 0.3
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
    print("\nTop 10 strongest input-output correlations:")
    flat_corr = correlation_matrix.unstack()
    strongest_corr = flat_corr[abs(flat_corr).sort_values(ascending=False).index][:10]
    
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
            save_path=str(output_dir / "input_output_correlation.png")
        )
        
        # Analyze correlations
        analyze_correlations(correlation_matrix)
        
        # Save correlation matrix to CSV
        correlation_matrix.to_csv(output_dir / "input_output_correlation.csv")
        print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()