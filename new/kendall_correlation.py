import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from correlation import load_data, calculate_enhanced_features

def calculate_kendall_correlation(data_path, genotype, parent_dir):
    """
    Calculate Kendall's Tau correlation coefficient between variables in the dataset.
    
    Args:
        data_path (str): Path to the CSV file containing the data
        genotype (str): Genotype to analyze ('P9RT', 'BPN', or 'P9LT')
        parent_dir (Path): Parent directory to save all correlation results
    """
    # Create genotype directory
    genotype_dir = parent_dir / genotype
    genotype_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data using correlation.py functions
    print(f"\nLoading and preprocessing data for {genotype} from: {data_path}")
    df = load_data(data_path, genotype)
    
    # Separate features and joint angles
    joint_cols = [col for col in df.columns if col.endswith(('_flex', '_rot', '_abduct'))]
    feature_cols = [col for col in df.columns if col not in joint_cols]
    
    print(f"\nAnalyzing correlations between {len(feature_cols)} features and {len(joint_cols)} joint angles")
    
    # Calculate Kendall's Tau correlation
    print("\nCalculating Kendall's Tau correlation coefficients...")
    correlation_matrix = df[feature_cols + joint_cols].corr(method='kendall')
    
    # Save full correlation matrix to CSV
    correlation_csv = genotype_dir / 'kendall_correlation_matrix.csv'
    correlation_matrix.to_csv(correlation_csv)
    print(f"Saved correlation matrix to: {correlation_csv}")
    
    # Create correlation heatmap for features vs joint angles
    feature_joint_correlation = correlation_matrix.loc[feature_cols, joint_cols]
    
    plt.figure(figsize=(24, 12))
    sns.heatmap(feature_joint_correlation, 
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.2f',
                vmin=-1,
                vmax=1,
                cbar_kws={"shrink": .8})
    
    plt.title(f"Kendall's Tau Correlation: Features vs Joint Angles ({genotype})")
    plt.xlabel('Joint Angles')
    plt.ylabel('Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save heatmap
    heatmap_path = genotype_dir / 'kendall_correlation_heatmap.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation heatmap to: {heatmap_path}")
    
    # Calculate and save significant correlations
    print("\nAnalyzing significant correlations...")
    significant_correlations = []
    
    for feature in feature_cols:
        for joint in joint_cols:
            correlation, p_value = stats.kendalltau(df[feature], df[joint], nan_policy='omit')
            if abs(correlation) > 0.3 and p_value < 0.05:  # Same threshold as Spearman
                significant_correlations.append({
                    'Feature': feature,
                    'Joint Angle': joint,
                    'Correlation': correlation,
                    'P-value': p_value
                })
    
    # Sort by absolute correlation strength
    significant_correlations.sort(key=lambda x: abs(x['Correlation']), reverse=True)
    
    # Save significant correlations to CSV
    if significant_correlations:
        sig_corr_df = pd.DataFrame(significant_correlations)
        sig_corr_path = genotype_dir / 'significant_correlations.csv'
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
        
        for i, corr in enumerate(significant_correlations[:20]):  # Top 20 correlations
            feature, joint = corr['Feature'], corr['Joint Angle']
            
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x=feature, y=joint, alpha=0.5)
            plt.title(f"{genotype}: Kendall's Tau: {corr['Correlation']:.3f} (p={corr['P-value']:.3e})")
            plt.xlabel(feature)
            plt.ylabel(joint)
            
            # Add trend line
            z = np.polyfit(df[feature], df[joint], 1)
            p = np.poly1d(z)
            plt.plot(df[feature], p(df[feature]), "r--", alpha=0.8)
            
            plt.tight_layout()
            scatter_path = genotype_dir / f'scatter_{i+1}_{feature}_vs_{joint}.png'
            plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Saved scatter plots to: {genotype_dir}")

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
        
        parent_dir = Path("kendall_correlation_results")
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
                correlation_matrix = calculate_kendall_correlation(file_path, genotype, output_dir)
                
                plot_correlation_heatmap(
                    correlation_matrix,
                    save_path=str(output_dir / f"kendall_correlation_{genotype}.png")
                )
                
                print(f"\nKendall Correlation Analysis for {genotype}:")
                analyze_correlations(correlation_matrix, output_dir=output_dir)
                
                correlation_matrix.to_csv(output_dir / f"kendall_correlation_matrix_{genotype}.csv")
                print(f"\nResults saved to {output_dir}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data file is available in one of the expected locations.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 