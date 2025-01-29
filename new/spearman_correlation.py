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
    
    # Separate features and joint angles
    joint_cols = [col for col in df.columns if col.endswith(('_flex', '_rot', '_abduct'))]
    feature_cols = [col for col in df.columns if col not in joint_cols]
    
    print(f"\nAnalyzing correlations between {len(feature_cols)} features and {len(joint_cols)} joint angles")
    
    # Calculate Spearman correlation
    print("\nCalculating Spearman correlation coefficients...")
    correlation_matrix = df[feature_cols + joint_cols].corr(method='spearman')
    
    # Save full correlation matrix to CSV
    correlation_csv = output_path / f'spearman_correlation_matrix_{genotype}.csv'
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
    
    plt.title(f"Spearman's Rank Correlation: Features vs Joint Angles ({genotype})")
    plt.xlabel('Joint Angles')
    plt.ylabel('Features')
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
            correlation, p_value = stats.spearmanr(df[feature], df[joint], nan_policy='omit')
            if abs(correlation) > 0.3 and p_value < 0.05:  # Adjusted threshold to 0.3
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
            feature, joint = corr['Feature'], corr['Joint Angle']
            
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x=feature, y=joint, alpha=0.5)
            plt.title(f"{genotype}: Spearman Correlation: {corr['Correlation']:.3f} (p={corr['P-value']:.3e})")
            plt.xlabel(feature)
            plt.ylabel(joint)
            
            # Add trend line
            z = np.polyfit(df[feature], df[joint], 1)
            p = np.poly1d(z)
            plt.plot(df[feature], p(df[feature]), "r--", alpha=0.8)
            
            plt.tight_layout()
            scatter_path = scatter_dir / f'scatter_{i+1}_{feature}_vs_{joint}.png'
            plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Saved scatter plots to: {scatter_dir}")

def main():
    """Main function to run the correlation analysis."""
    # Set up paths
    data_path = "/Users/anivenkat/Downloads/BPN_P9LT_P9RT_flyCoords.csv"
    output_dir = "spearman_correlation_results"
    
    # Define genotypes to analyze
    genotypes = ['P9RT', 'BPN', 'P9LT']
    
    # Run analysis for each genotype
    try:
        print("Starting Spearman correlation analysis...")
        for genotype in genotypes:
            print(f"\nAnalyzing {genotype} genotype...")
            calculate_spearman_correlation(data_path, genotype, output_dir)
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 