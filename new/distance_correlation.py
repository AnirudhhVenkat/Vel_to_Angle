import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
from correlation import load_data, calculate_enhanced_features

def distance_correlation(X, Y):
    """
    Calculate the distance correlation between two variables.
    Distance correlation is zero if and only if the variables are independent.
    
    Args:
        X: First variable (n_samples,)
        Y: Second variable (n_samples,)
    
    Returns:
        Distance correlation coefficient (between 0 and 1)
    """
    # Convert to numpy arrays
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    # Calculate pairwise distances
    X_dist = squareform(pdist(X.reshape(-1, 1)))
    Y_dist = squareform(pdist(Y.reshape(-1, 1)))
    
    # Double center the distance matrices
    n = len(X)
    H = np.eye(n) - np.ones((n, n)) / n
    X_centered = H @ X_dist @ H
    Y_centered = H @ Y_dist @ H
    
    # Calculate distance covariance and variances
    dCov = np.sqrt(np.sum(X_centered * Y_centered)) / n
    dVarX = np.sqrt(np.sum(X_centered * X_centered)) / n
    dVarY = np.sqrt(np.sum(Y_centered * Y_centered)) / n
    
    # Calculate distance correlation
    if dVarX * dVarY == 0:
        return 0
    else:
        return dCov / np.sqrt(dVarX * dVarY)

def calculate_distance_correlation(data_path, genotype, parent_dir):
    """
    Calculate distance correlation between variables in the dataset.
    Distance correlation can detect both linear and nonlinear relationships.
    
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
    
    print(f"\nAnalyzing distance correlations between {len(feature_cols)} features and {len(joint_cols)} joint angles")
    
    # Calculate distance correlation matrix
    print("\nCalculating distance correlation coefficients...")
    dcor_matrix = np.zeros((len(feature_cols), len(joint_cols)))
    
    for i, feature in enumerate(feature_cols):
        for j, joint in enumerate(joint_cols):
            dcor = distance_correlation(df[feature], df[joint])
            dcor_matrix[i, j] = dcor
    
    # Create DataFrame for the distance correlation matrix
    dcor_df = pd.DataFrame(dcor_matrix, index=feature_cols, columns=joint_cols)
    
    # Save full distance correlation matrix to CSV
    dcor_csv = genotype_dir / 'distance_correlation_matrix.csv'
    dcor_df.to_csv(dcor_csv)
    print(f"Saved distance correlation matrix to: {dcor_csv}")
    
    # Create distance correlation heatmap
    plt.figure(figsize=(30, 12))
    sns.heatmap(dcor_df, 
                annot=True,
                cmap='viridis',  # Different colormap since distance correlation is always non-negative
                fmt='.3f',
                vmin=0,  # Distance correlation is always between 0 and 1
                vmax=1,
                cbar_kws={"shrink": .8})
    
    plt.title(f"Distance Correlation: Features vs Joint Angles ({genotype})")
    plt.xlabel('Joint Angles')
    plt.ylabel('Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save heatmap
    heatmap_path = genotype_dir / 'distance_correlation_heatmap.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation heatmap to: {heatmap_path}")
    
    # Find significant relationships
    # For distance correlation, we'll use a higher threshold since it's always non-negative
    dcor_threshold = 0.5  # Adjusted threshold for distance correlation
    significant_relationships = []
    
    for feature in feature_cols:
        for joint in joint_cols:
            dcor = dcor_df.loc[feature, joint]
            if dcor > dcor_threshold:
                significant_relationships.append({
                    'Feature': feature,
                    'Joint Angle': joint,
                    'Distance Correlation': dcor
                })
    
    # Sort by distance correlation value
    significant_relationships.sort(key=lambda x: x['Distance Correlation'], reverse=True)
    
    # Save significant relationships to CSV
    if significant_relationships:
        sig_dcor_df = pd.DataFrame(significant_relationships)
        sig_dcor_path = genotype_dir / 'significant_relationships.csv'
        sig_dcor_df.to_csv(sig_dcor_path, index=False)
        print(f"Saved significant relationships to: {sig_dcor_path}")
        
        # Print top relationships
        print(f"\nTop 20 strongest relationships for {genotype}:")
        print(sig_dcor_df.head(20).to_string(index=False))
    else:
        print(f"\nNo significant relationships found for {genotype}.")
    
    # Create scatter plots for top relationships
    if significant_relationships:
        print("\nGenerating scatter plots for top relationships...")
        
        for i, rel in enumerate(significant_relationships[:20]):  # Top 20 relationships
            feature, joint = rel['Feature'], rel['Joint Angle']
            
            plt.figure(figsize=(10, 6))
            
            # Create scatter plot with hexbin for better density visualization
            plt.hexbin(df[feature], df[joint], gridsize=30, cmap='viridis')
            plt.colorbar(label='Count')
            
            plt.title(f"{genotype}: Distance Correlation: {rel['Distance Correlation']:.3f}")
            plt.xlabel(feature)
            plt.ylabel(joint)
            
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
        
        parent_dir = Path("distance_correlation_results")
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
                dcor_matrix = compute_distance_correlation(df)
                
                plot_dcor_heatmap(
                    dcor_matrix,
                    save_path=str(output_dir / f"distance_correlation_{genotype}.png")
                )
                
                print(f"\nDistance Correlation Analysis for {genotype}:")
                analyze_dcor(dcor_matrix, output_dir=output_dir)
                
                dcor_matrix.to_csv(output_dir / f"distance_correlation_matrix_{genotype}.csv")
                print(f"\nResults saved to {output_dir}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data file is available in one of the expected locations.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 