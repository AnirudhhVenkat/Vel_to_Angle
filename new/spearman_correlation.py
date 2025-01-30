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
    
    for window in windows:
        for col in tag_cols:
            ma_col = f"{col}_ma{window}"
            df[ma_col] = df[col].rolling(window=window, center=True).mean()
    
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
    
    # Calculate correlations between each input-output pair
    for feature in feature_cols:
        for joint in joint_cols:
            try:
                # Get the data as numpy arrays, ensuring we get 1D arrays
                x = df_combined[feature].to_numpy().ravel()
                y = df_combined[joint].to_numpy().ravel()
                
                # Print debug info if shapes don't match
                if len(x) != len(y):
                    print(f"\nShape mismatch for {feature} vs {joint}:")
                    print(f"Shape of {feature}: {x.shape}")
                    print(f"Shape of {joint}: {y.shape}")
                    continue
                
                # Remove any NaN values
                mask = ~(np.isnan(x) | np.isnan(y))
                x = x[mask]
                y = y[mask]
                
                # Calculate Spearman correlation using scipy stats
                if len(x) > 0 and len(y) > 0:
                    correlation, _ = stats.spearmanr(x, y)
                else:
                    correlation = np.nan
                correlation_matrix.loc[feature, joint] = correlation
                
            except Exception as e:
                print(f"\nError calculating correlation between {feature} and {joint}: {e}")
                correlation_matrix.loc[feature, joint] = np.nan
    
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
    
    for feature in feature_cols:
        for joint in joint_cols:
            try:
                # Get the data as numpy arrays, ensuring we get 1D arrays
                x = df_combined[feature].to_numpy().ravel()
                y = df_combined[joint].to_numpy().ravel()
                
                if len(x) != len(y):
                    continue
                
                # Remove any NaN values
                mask = ~(np.isnan(x) | np.isnan(y))
                x = x[mask]
                y = y[mask]
                
                # Calculate correlation and p-value
                if len(x) > 0 and len(y) > 0:
                    correlation, p_value = stats.spearmanr(x, y)
                    if abs(correlation) > 0.3 and p_value < 0.05:  # Adjusted threshold to 0.3
                        significant_correlations.append({
                            'Feature': feature,
                            'Joint Angle': joint,
                            'Correlation': correlation,
                            'P-value': p_value
                        })
            except Exception as e:
                print(f"\nError analyzing correlation between {feature} and {joint}: {e}")
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
    
    # Create scatter plots for top correlations
    if significant_correlations:
        print("\nGenerating scatter plots for top correlations...")
        scatter_dir = output_path / 'scatter_plots'
        scatter_dir.mkdir(exist_ok=True)
        
        for i, corr in enumerate(significant_correlations[:20]):  # Top 20 correlations
            try:
                feature, joint = corr['Feature'], corr['Joint Angle']
                
                plt.figure(figsize=(10, 6))
                x = df_combined[feature].to_numpy()
                y = df_combined[joint].to_numpy()
                
                # Remove NaN values for plotting
                mask = ~(np.isnan(x) | np.isnan(y))
                x = x[mask]
                y = y[mask]
                
                plt.scatter(x, y, alpha=0.5)
                plt.title(f"{genotype}: Spearman Correlation: {corr['Correlation']:.3f} (p={corr['P-value']:.3e})")
                plt.xlabel(feature)
                plt.ylabel(joint)
                
                # Add trend line
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                plt.plot(x, p(x), "r--", alpha=0.8)
                
                plt.tight_layout()
                scatter_path = scatter_dir / f'scatter_{i+1}_{feature}_vs_{joint}.png'
                plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"\nError creating scatter plot for {feature} vs {joint}: {e}")
                continue
        
        print(f"Saved scatter plots to: {scatter_dir}")
    
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