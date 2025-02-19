import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from correlation import load_data, calculate_enhanced_features

def calculate_spearman_correlation(data_path, genotype, output_dir='spearman_correlation_analysis'):
    """
    Calculate Spearman's rank correlation coefficient between variables in the dataset.
    
    Args:
        data_path (str): Path to the CSV file containing the data
        genotype (str): Genotype to analyze ('P9RT', 'BPN', or 'P9LT')
        output_dir (str): Directory to save the correlation results and plots
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir) / genotype
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data using correlation.py functions
    print(f"\nLoading and preprocessing data for {genotype} from: {data_path}")
    df = load_data(data_path, genotype)
    
    # Calculate enhanced features
    print("\nCalculating enhanced features (positions, velocities, etc.)...")
    enhanced_features = calculate_enhanced_features(df)
    
    # Calculate moving averages for TaG positions
    print("\nCalculating moving averages for TaG positions...")
    tag_cols = [col for col in df.columns if 'TaG' in col]
    windows = [5, 10, 20]  # Window sizes for moving averages
    
    # Process each trial separately to avoid boundary issues
    trial_size = 600  # After filtering (1000 - 400 = 600)
    num_trials = len(df) // trial_size
    
    for window in windows:
        for col in tag_cols:
            ma_values = []
            for trial in range(num_trials):
                start_idx = trial * trial_size
                end_idx = start_idx + trial_size
                trial_data = df[col].iloc[start_idx:end_idx]
                # Calculate moving average within the trial
                ma = trial_data.rolling(window=window, center=True, min_periods=1).mean()
                ma_values.extend(ma.tolist())
            df[f"{col}_ma{window}"] = ma_values
    
    # Combine original data with enhanced features
    df_combined = pd.concat([df, enhanced_features], axis=1)
    
    # Print shape information for debugging
    print(f"\nShape of original data: {df.shape}")
    print(f"Shape of enhanced features: {enhanced_features.shape}")
    print(f"Shape of combined data: {df_combined.shape}")
    
    # Separate features and joint angles
    joint_cols = [col for col in df_combined.columns if col.endswith(('_flex', '_rot', '_abduct'))]
    
    # Get all input features: TaG points, velocities, positions, accelerations, etc.
    # Make sure to only get unique columns
    all_features = [col for col in df_combined.select_dtypes(include=[np.number]).columns 
                   if any(x in col.lower() for x in ['tag', 'vel', 'pos', 'acc', 'jerk', 'magnitude', 'ma'])]
    feature_cols = list(dict.fromkeys([col for col in all_features 
                                     if col not in joint_cols 
                                     and col != 'genotype']))
    
    print(f"\nAnalyzing correlations between {len(feature_cols)} features and {len(joint_cols)} joint angles")
    print("\nFeatures being analyzed:")
    for feat in feature_cols:
        print(f"  {feat}")
    
    # Calculate Spearman correlation
    print("\nCalculating Spearman correlation coefficients...")
    # Initialize correlation matrix
    correlation_matrix = pd.DataFrame(
        np.zeros((len(feature_cols), len(joint_cols))),
        index=feature_cols,
        columns=joint_cols
    )
    
    # Process each trial separately
    for trial in range(num_trials):
        start_idx = trial * trial_size
        end_idx = start_idx + trial_size
        
        # Calculate correlations between each input-output pair
        for feature in feature_cols:
            for joint in joint_cols:
                try:
                    # Get the data for this trial
                    x = df_combined[feature].iloc[start_idx:end_idx].to_numpy()
                    y = df_combined[joint].iloc[start_idx:end_idx].to_numpy()
                    
                    # Remove any NaN values
                    mask = ~(np.isnan(x) | np.isnan(y))
                    x = x[mask]
                    y = y[mask]
                    
                    # Calculate Spearman correlation if we have enough data
                    if len(x) > 0 and len(y) > 0:
                        correlation, _ = stats.spearmanr(x, y)
                        # Accumulate correlations (we'll take the mean later)
                        if np.isnan(correlation_matrix.loc[feature, joint]):
                            correlation_matrix.loc[feature, joint] = correlation
                        else:
                            correlation_matrix.loc[feature, joint] += correlation
                    
                except Exception as e:
                    print(f"\nError calculating correlation between {feature} and {joint} for trial {trial}: {e}")
                    continue
    
    # Take the mean of correlations across trials
    correlation_matrix = correlation_matrix / num_trials
    
    # Save full correlation matrix to CSV
    correlation_csv = output_path / f'spearman_correlation_matrix_{genotype}.csv'
    correlation_matrix.to_csv(correlation_csv)
    print(f"Saved correlation matrix to: {correlation_csv}")
    
    # Create correlation heatmap for features vs joint angles
    plt.figure(figsize=(24, 12))
    sns.heatmap(correlation_matrix, 
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.2f',
                vmin=-1,
                vmax=1,
                cbar_kws={"shrink": .8})
    
    plt.title(f"Spearman's Rank Correlation: All Features vs Joint Angles ({genotype})")
    plt.xlabel('Joint Angles')
    plt.ylabel('Features (TaG, Velocities, Positions, etc.)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save heatmap
    heatmap_path = output_path / f'spearman_correlation_heatmap_{genotype}.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation heatmap to: {heatmap_path}")
    
    # Calculate and save significant correlations
    print("\nAnalyzing significant correlations...")
    significant_correlations = []
    
    # Process each trial separately
    for trial in range(num_trials):
        start_idx = trial * trial_size
        end_idx = start_idx + trial_size
        
        for feature in feature_cols:
            for joint in joint_cols:
                try:
                    # Get the data for this trial
                    x = df_combined[feature].iloc[start_idx:end_idx].to_numpy()
                    y = df_combined[joint].iloc[start_idx:end_idx].to_numpy()
                    
                    # Remove any NaN values
                    mask = ~(np.isnan(x) | np.isnan(y))
                    x = x[mask]
                    y = y[mask]
                    
                    # Calculate correlation and p-value if we have enough data
                    if len(x) > 0 and len(y) > 0:
                        correlation, p_value = stats.spearmanr(x, y)
                        if abs(correlation) > 0.3 and p_value < 0.05:  # Adjusted threshold to 0.3
                            significant_correlations.append({
                                'Feature': feature,
                                'Joint Angle': joint,
                                'Correlation': correlation,
                                'P-value': p_value,
                                'Trial': trial + 1
                            })
                except Exception as e:
                    print(f"\nError analyzing correlation between {feature} and {joint} for trial {trial}: {e}")
                    continue
    
    # Sort by absolute correlation strength
    significant_correlations.sort(key=lambda x: abs(x['Correlation']), reverse=True)
    
    # Save significant correlations to CSV
    if significant_correlations:
        sig_corr_df = pd.DataFrame(significant_correlations)
        sig_corr_path = output_path / f'significant_correlations_{genotype}.csv'
        sig_corr_df.to_csv(sig_corr_path, index=False)
        print(f"Saved significant correlations to: {sig_corr_path}")
        
        # Print top correlations
        print(f"\nTop 20 strongest correlations for {genotype}:")
        print(sig_corr_df.head(20).to_string(index=False))
    else:
        print(f"\nNo significant correlations found for {genotype}.")
    
    return correlation_matrix

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
        
        parent_dir = Path("spearman_correlation_results")
        parent_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each genotype
        genotypes = ['P9RT', 'BPN', 'P9LT']
        
        for genotype in genotypes:
            print(f"\nProcessing {genotype} genotype...")
            output_dir = parent_dir / genotype
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Calculate correlations and generate plots
            calculate_spearman_correlation(file_path, genotype, output_dir)
            print(f"\nResults saved to {output_dir}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data file is available in one of the expected locations.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 