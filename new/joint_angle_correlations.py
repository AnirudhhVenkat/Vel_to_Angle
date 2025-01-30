import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from tqdm import tqdm

def load_data(data_path, genotype):
    """Load data for a specific genotype."""
    df = pd.read_csv(data_path)
    return df[df['genotype'] == genotype]

def analyze_joint_correlations(data_path, genotype, parent_dir):
    """
    Calculate Spearman correlation between joint angles.
    
    Args:
        data_path (str): Path to the CSV file containing the data
        genotype (str): Genotype to analyze ('P9RT', 'BPN', or 'P9LT')
        parent_dir (Path): Parent directory to save correlation results
    """
    # Create output directory
    output_dir = parent_dir / genotype
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\nLoading data for {genotype} from: {data_path}")
    df = load_data(data_path, genotype)
    
    # Get joint angle columns
    joint_cols = [col for col in df.columns if col.endswith(('_flex', '_rot', '_abduct'))]
    
    print(f"\nAnalyzing correlations between {len(joint_cols)} joint angles")
    print("\nJoint angles being analyzed:")
    for joint in joint_cols:
        print(f"  {joint}")
    
    # Calculate correlation matrix
    print("\nCalculating Spearman correlations...")
    correlation_matrix = df[joint_cols].corr(method='spearman')
    
    # Save correlation matrix
    correlation_matrix.to_csv(output_dir / 'joint_angle_correlations.csv')
    
    # Create heatmap
    plt.figure(figsize=(20, 20))
    sns.heatmap(correlation_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.2f',
                vmin=-1,
                vmax=1,
                cbar_kws={"shrink": .8})
    
    plt.title(f"Joint Angle Correlation Matrix ({genotype})")
    plt.xlabel('Joint Angles')
    plt.ylabel('Joint Angles')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'joint_angle_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find significant correlations
    print("\nAnalyzing significant correlations...")
    significant_correlations = []
    
    # Get upper triangle indices (to avoid duplicates)
    upper_triangle = np.triu_indices_from(correlation_matrix, k=1)
    
    # Find significant correlations from upper triangle
    for i, j in zip(*upper_triangle):
        corr = correlation_matrix.iloc[i, j]
        if abs(corr) > 0.3:  # Same threshold as other correlation analyses
            significant_correlations.append({
                'Joint Angle 1': correlation_matrix.index[i],
                'Joint Angle 2': correlation_matrix.columns[j],
                'Correlation': corr
            })
    
    # Sort by absolute correlation strength
    significant_correlations.sort(key=lambda x: abs(x['Correlation']), reverse=True)
    
    # Save significant correlations
    if significant_correlations:
        sig_corr_df = pd.DataFrame(significant_correlations)
        sig_corr_df.to_csv(output_dir / 'significant_joint_correlations.csv', index=False)
        
        # Print top correlations
        print(f"\nTop 20 strongest correlations for {genotype}:")
        print(sig_corr_df.head(20).to_string(index=False))
        
        # Create scatter plots for top correlations
        print("\nGenerating scatter plots for top correlations...")
        scatter_dir = output_dir / 'scatter_plots'
        scatter_dir.mkdir(exist_ok=True)
        
        for i, corr in tqdm(enumerate(significant_correlations[:20]), 
                           total=min(20, len(significant_correlations)),
                           desc="Creating scatter plots"):
            joint1 = corr['Joint Angle 1']
            joint2 = corr['Joint Angle 2']
            correlation = corr['Correlation']
            
            plt.figure(figsize=(10, 6))
            plt.scatter(df[joint1], df[joint2], alpha=0.5)
            plt.title(f"{genotype}\nCorrelation: {correlation:.3f}")
            plt.xlabel(joint1)
            plt.ylabel(joint2)
            
            # Add trend line
            z = np.polyfit(df[joint1], df[joint2], 1)
            p = np.poly1d(z)
            plt.plot(df[joint1], p(df[joint1]), "r--", alpha=0.8)
            
            plt.tight_layout()
            plt.savefig(scatter_dir / f'scatter_{i+1}_{joint1}_vs_{joint2}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    else:
        print(f"\nNo significant correlations found for {genotype}.")

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
        
        parent_dir = Path("joint_angle_correlation_results")
        parent_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each genotype
        genotypes = ['P9RT', 'BPN', 'P9LT']
        
        for genotype in genotypes:
            print(f"\nProcessing {genotype} genotype...")
            analyze_joint_correlations(file_path, genotype, parent_dir)
            print(f"\nResults saved to {parent_dir}/{genotype}/")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data file is available in one of the expected locations.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 