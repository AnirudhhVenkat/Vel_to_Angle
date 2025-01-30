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

def calculate_enhanced_features(df):
    """Calculate enhanced features including integrated positions."""
    enhanced_features = pd.DataFrame()
    
    # Calculate positions by integrating velocities
    enhanced_features['x_pos'] = df['x_vel'].cumsum()
    enhanced_features['y_pos'] = df['y_vel'].cumsum()
    enhanced_features['z_pos'] = df['z_vel'].cumsum()
    
    return enhanced_features

def analyze_xyz_correlations(data_path, genotype, parent_dir):
    """
    Calculate Spearman correlation between all variables ending in _x, _y, or _z,
    and include joint angles in the analysis.
    
    Args:
        data_path (str): Path to the CSV file containing the data
        genotype (str): Genotype to analyze ('P9RT', 'BPN', or 'P9LT')
        parent_dir (Path): Parent directory to save correlation results
    """
    # Create output directory
    output_dir = parent_dir / genotype
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    print(f"\nLoading and preprocessing data for {genotype} from: {data_path}")
    df = load_data(data_path, genotype)
    
    # Calculate enhanced features
    print("\nCalculating enhanced features including integrated positions...")
    enhanced_features = calculate_enhanced_features(df)
    
    # Combine enhanced features with original data
    combined_df = pd.concat([df, enhanced_features], axis=1)
    
    # Get all columns ending in _x, _y, or _z
    x_cols = [col for col in combined_df.columns if col.endswith('_x')]
    y_cols = [col for col in combined_df.columns if col.endswith('_y')]
    z_cols = [col for col in combined_df.columns if col.endswith('_z')]
    
    # Get joint angle columns (assuming they end with 'angle')
    joint_cols = [col for col in combined_df.columns if col.endswith('angle')]
    
    # Combine all columns for analysis
    all_cols = sorted(x_cols + y_cols + z_cols + joint_cols)
    
    print("\nAnalyzing correlations between:")
    print(f"- {len(x_cols)} variables ending in _x")
    print(f"- {len(y_cols)} variables ending in _y")
    print(f"- {len(z_cols)} variables ending in _z")
    print(f"- {len(joint_cols)} joint angles")
    
    # Create correlation matrix using pandas' built-in method
    print("\nCalculating Spearman correlations...")
    correlation_matrix = combined_df[all_cols].corr(method='spearman')
    
    # Save correlation matrix
    correlation_matrix.to_csv(output_dir / 'xyz_joint_correlations.csv')
    
    # Create heatmap
    plt.figure(figsize=(30, 30))
    sns.heatmap(correlation_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.2f',
                vmin=-1,
                vmax=1,
                cbar_kws={"shrink": .8})
    
    plt.title(f"Correlation Matrix of XYZ Variables and Joint Angles ({genotype})")
    plt.xlabel('Variables')
    plt.ylabel('Variables')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'xyz_joint_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create separate heatmap for joint angles vs xyz variables
    xyz_joint_matrix = correlation_matrix.loc[x_cols + y_cols + z_cols, joint_cols]
    
    plt.figure(figsize=(20, 30))
    sns.heatmap(xyz_joint_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.2f',
                vmin=-1,
                vmax=1,
                cbar_kws={"shrink": .8})
    
    plt.title(f"XYZ Variables vs Joint Angles Correlation ({genotype})")
    plt.xlabel('Joint Angles')
    plt.ylabel('XYZ Variables')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'xyz_vs_joints_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find significant correlations
    print("\nAnalyzing significant correlations...")
    significant_correlations = []
    
    # Get upper triangle indices
    upper_triangle = np.triu_indices_from(correlation_matrix, k=1)
    
    # Find significant correlations from upper triangle
    for i, j in zip(*upper_triangle):
        corr = correlation_matrix.iloc[i, j]
        if abs(corr) > 0.3:  # Same threshold as other correlation analyses
            significant_correlations.append({
                'Variable 1': correlation_matrix.index[i],
                'Variable 2': correlation_matrix.columns[j],
                'Correlation': corr,
                'Type': 'Joint-Joint' if correlation_matrix.index[i] in joint_cols and correlation_matrix.columns[j] in joint_cols else
                       'XYZ-Joint' if (correlation_matrix.index[i] in joint_cols) != (correlation_matrix.columns[j] in joint_cols) else
                       'XYZ-XYZ'
            })
    
    # Sort by absolute correlation strength
    significant_correlations.sort(key=lambda x: abs(x['Correlation']), reverse=True)
    
    # Save significant correlations
    if significant_correlations:
        sig_corr_df = pd.DataFrame(significant_correlations)
        sig_corr_df.to_csv(output_dir / 'significant_correlations.csv', index=False)
        
        # Print top correlations by type
        for corr_type in ['XYZ-Joint', 'XYZ-XYZ', 'Joint-Joint']:
            type_correlations = sig_corr_df[sig_corr_df['Type'] == corr_type]
            if not type_correlations.empty:
                print(f"\nTop 10 strongest {corr_type} correlations for {genotype}:")
                print(type_correlations.head(10).to_string(index=False))
        
        # Create scatter plots for top correlations
        print("\nGenerating scatter plots for top correlations...")
        scatter_dir = output_dir / 'scatter_plots'
        scatter_dir.mkdir(exist_ok=True)
        
        # Create scatter plots for top 20 correlations of each type
        for corr_type in ['XYZ-Joint', 'XYZ-XYZ', 'Joint-Joint']:
            type_correlations = sig_corr_df[sig_corr_df['Type'] == corr_type]
            
            for i, corr in tqdm(enumerate(type_correlations.head(20).iterrows()), 
                               total=min(20, len(type_correlations)),
                               desc=f"Creating {corr_type} scatter plots"):
                _, row = corr  # Unpack the index and row from iterrows()
                var1 = row['Variable 1']
                var2 = row['Variable 2']
                correlation = row['Correlation']
                
                plt.figure(figsize=(10, 6))
                
                # Get data from appropriate source
                x = enhanced_features[var1] if var1 in enhanced_features.columns else combined_df[var1]
                y = enhanced_features[var2] if var2 in enhanced_features.columns else combined_df[var2]
                
                plt.scatter(x, y, alpha=0.5)
                plt.title(f"{genotype}: {corr_type}\nCorrelation: {correlation:.3f}")
                plt.xlabel(var1)
                plt.ylabel(var2)
                
                # Add trend line
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                plt.plot(x, p(x), "r--", alpha=0.8)
                
                plt.tight_layout()
                plt.savefig(scatter_dir / f'{corr_type}_scatter_{i+1}_{var1}_vs_{var2}.png', 
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
        
        parent_dir = Path("xyz_correlation_results")
        parent_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each genotype
        genotypes = ['P9RT', 'BPN', 'P9LT']
        
        for genotype in genotypes:
            print(f"\nProcessing {genotype} genotype...")
            output_dir = parent_dir / genotype
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Calculate correlations and generate plots
            analyze_xyz_correlations(file_path, genotype, parent_dir)
            print(f"\nResults saved to {output_dir}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data file is available in one of the expected locations.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 