import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from pathlib import Path
from correlation import load_data, calculate_enhanced_features

def calculate_mutual_information(data_path, genotype, parent_dir):
    """
    Calculate mutual information between variables in the dataset.
    Mutual information measures how much information one variable provides about another,
    capturing both linear and non-linear relationships.
    
    Args:
        data_path (str): Path to the CSV file containing the data
        genotype (str): Genotype to analyze ('P9RT', 'BPN', or 'P9LT')
        parent_dir (Path): Parent directory to save all mutual information results
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
    
    print(f"\nAnalyzing mutual information between {len(feature_cols)} features and {len(joint_cols)} joint angles")
    
    # Calculate mutual information matrix
    print("\nCalculating mutual information scores...")
    mi_matrix = np.zeros((len(feature_cols), len(joint_cols)))
    
    for i, joint in enumerate(joint_cols):
        # Calculate mutual information between all features and this joint angle
        mi_scores = mutual_info_regression(df[feature_cols], df[joint], random_state=42)
        mi_matrix[:, i] = mi_scores
    
    # Create DataFrame for the mutual information matrix
    mi_df = pd.DataFrame(mi_matrix, index=feature_cols, columns=joint_cols)
    
    # Save full mutual information matrix to CSV
    mi_csv = genotype_dir / 'mutual_information_matrix.csv'
    mi_df.to_csv(mi_csv)
    print(f"Saved mutual information matrix to: {mi_csv}")
    
    # Create mutual information heatmap
    plt.figure(figsize=(24, 12))
    sns.heatmap(mi_df, 
                annot=True,
                cmap='viridis',  # Different colormap for MI (always positive)
                fmt='.3f',
                vmin=0,  # MI is always non-negative
                cbar_kws={"shrink": .8})
    
    plt.title(f"Mutual Information: Features vs Joint Angles ({genotype})")
    plt.xlabel('Joint Angles')
    plt.ylabel('Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save heatmap
    heatmap_path = genotype_dir / 'mutual_information_heatmap.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved mutual information heatmap to: {heatmap_path}")
    
    # Find significant relationships
    # For MI, we'll use the mean + 1 standard deviation as the threshold
    mi_threshold = mi_matrix.mean() + mi_matrix.std()
    significant_relationships = []
    
    for feature in feature_cols:
        for joint in joint_cols:
            mi_score = mi_df.loc[feature, joint]
            if mi_score > mi_threshold:
                significant_relationships.append({
                    'Feature': feature,
                    'Joint Angle': joint,
                    'Mutual Information': mi_score
                })
    
    # Sort by mutual information score
    significant_relationships.sort(key=lambda x: x['Mutual Information'], reverse=True)
    
    # Save significant relationships to CSV
    if significant_relationships:
        sig_mi_df = pd.DataFrame(significant_relationships)
        sig_mi_path = genotype_dir / 'significant_relationships.csv'
        sig_mi_df.to_csv(sig_mi_path, index=False)
        print(f"Saved significant relationships to: {sig_mi_path}")
        
        # Print top relationships
        print(f"\nTop 20 strongest relationships for {genotype}:")
        print(sig_mi_df.head(20).to_string(index=False))
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
            
            plt.title(f"{genotype}: MI Score: {rel['Mutual Information']:.3f}")
            plt.xlabel(feature)
            plt.ylabel(joint)
            
            plt.tight_layout()
            scatter_path = genotype_dir / f'scatter_{i+1}_{feature}_vs_{joint}.png'
            plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Saved scatter plots to: {genotype_dir}")

def main():
    """Main function to run the mutual information analysis."""
    # Set up paths
    data_path = "/Users/anivenkat/Downloads/BPN_P9LT_P9RT_flyCoords.csv"
    parent_dir = Path("mutual_info_results")
    parent_dir.mkdir(exist_ok=True)
    
    # Define genotypes to analyze
    genotypes = ['P9RT', 'BPN', 'P9LT']
    
    # Run analysis for each genotype
    try:
        print("Starting mutual information analysis...")
        for genotype in genotypes:
            print(f"\nAnalyzing {genotype} genotype...")
            calculate_mutual_information(data_path, genotype, parent_dir)
        
        # Create a summary file
        with open(parent_dir / 'analysis_summary.txt', 'w') as f:
            f.write("Mutual Information Analysis Summary\n")
            f.write("==================================\n\n")
            f.write("Analysis completed for the following genotypes:\n")
            for genotype in genotypes:
                f.write(f"- {genotype}\n")
            f.write("\nEach genotype folder contains:\n")
            f.write("1. mutual_information_matrix.csv: Full MI scores matrix\n")
            f.write("2. mutual_information_heatmap.png: MI scores heatmap\n")
            f.write("3. significant_relationships.csv: List of significant relationships\n")
            f.write("4. scatter_*.png: Hexbin plots for top relationships\n")
            f.write("\nNote: Mutual information measures how much information one variable\n")
            f.write("provides about another, capturing both linear and non-linear relationships.\n")
            f.write("Higher scores indicate stronger relationships (range: 0 to inf).\n")
        
        print("\nAnalysis completed successfully!")
        print(f"All results have been saved in: {parent_dir}")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 