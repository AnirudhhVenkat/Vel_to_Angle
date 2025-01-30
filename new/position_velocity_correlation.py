import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from correlation import load_data, calculate_enhanced_features

def calculate_position_velocity_correlation(data_path, genotype, parent_dir):
    """
    Calculate Spearman correlation between integrated positions and TaG points/velocities.
    
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
    
    # Calculate enhanced features to get integrated positions
    print("\nCalculating enhanced features including integrated positions...")
    enhanced_features = calculate_enhanced_features(df)
    
    # Get integrated position and velocity columns
    integrated_pos_cols = ['x_pos', 'y_pos', 'z_pos']
    velocity_cols = ['x_vel', 'y_vel', 'z_vel']
    
    # Get TaG position and velocity columns
    tag_pos_cols = [col for col in df.columns if 'TaG' in col]
    tag_vel_cols = [col for col in df.columns if 'Vel' in col]
    
    print("\nAnalyzing correlations between:")
    print(f"- {len(integrated_pos_cols)} integrated positions")
    print(f"- {len(velocity_cols)} velocities")
    print(f"- {len(tag_pos_cols)} TaG positions")
    print(f"- {len(tag_vel_cols)} TaG velocities")
    
    # Create correlation matrix
    print("\nCalculating Spearman correlations...")
    
    # Combine all variables
    xyz_vars = integrated_pos_cols + velocity_cols
    tag_vars = tag_pos_cols + tag_vel_cols
    
    # Create correlation matrix
    correlation_matrix = pd.DataFrame(
        np.zeros((len(xyz_vars), len(tag_vars))),
        index=xyz_vars,
        columns=tag_vars
    )
    
    # Calculate correlations
    for xyz_var in xyz_vars:
        # Get data (either from enhanced_features or original df)
        if xyz_var in enhanced_features.columns:
            data = enhanced_features[xyz_var]
        else:
            data = df[xyz_var]
        
        # Correlate with TaG measurements
        for tag_var in tag_vars:
            correlation, _ = stats.spearmanr(data, df[tag_var], nan_policy='omit')
            correlation_matrix.loc[xyz_var, tag_var] = correlation
    
    # Save correlation matrix
    correlation_matrix.to_csv(output_dir / 'xyz_tag_correlations.csv')
    
    # Create heatmap
    plt.figure(figsize=(30, 12))
    sns.heatmap(correlation_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.2f',
                vmin=-1,
                vmax=1,
                cbar_kws={"shrink": .8})
    
    plt.title(f"XYZ Positions/Velocities vs TaG Measurements Correlation ({genotype})")
    plt.xlabel('TaG Positions and Velocities')
    plt.ylabel('XYZ Positions and Velocities')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'xyz_tag_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find significant correlations
    print("\nAnalyzing significant correlations...")
    significant_correlations = []
    
    # Check correlations
    for xyz_var in xyz_vars:
        for tag_var in tag_vars:
            corr = correlation_matrix.loc[xyz_var, tag_var]
            if abs(corr) > 0.3:  # Same threshold as other correlation analyses
                significant_correlations.append({
                    'XYZ Variable': xyz_var,
                    'TaG Variable': tag_var,
                    'Correlation': corr
                })
    
    # Sort by absolute correlation strength
    significant_correlations.sort(key=lambda x: abs(x['Correlation']), reverse=True)
    
    # Save significant correlations
    if significant_correlations:
        sig_corr_df = pd.DataFrame(significant_correlations)
        sig_corr_df.to_csv(output_dir / 'significant_correlations.csv', index=False)
        
        # Print top correlations
        print(f"\nTop 20 strongest correlations for {genotype}:")
        print(sig_corr_df.head(20).to_string(index=False))
        
        # Create scatter plots for top correlations
        print("\nGenerating scatter plots for top correlations...")
        scatter_dir = output_dir / 'scatter_plots'
        scatter_dir.mkdir(exist_ok=True)
        
        for i, corr in enumerate(significant_correlations[:20]):
            xyz_var = corr['XYZ Variable']
            tag_var = corr['TaG Variable']
            
            plt.figure(figsize=(10, 6))
            
            # Get correct data source for each variable
            x = enhanced_features[xyz_var] if xyz_var in enhanced_features.columns else df[xyz_var]
            y = df[tag_var]
            
            plt.scatter(x, y, alpha=0.5)
            plt.title(f"{genotype}: Correlation: {corr['Correlation']:.3f}")
            plt.xlabel(xyz_var)
            plt.ylabel(tag_var)
            
            # Add trend line
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(x, p(x), "r--", alpha=0.8)
            
            plt.tight_layout()
            plt.savefig(scatter_dir / f'scatter_{i+1}_{xyz_var}_vs_{tag_var}.png', 
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
        
        parent_dir = Path("position_velocity_correlation_results")
        parent_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each genotype
        genotypes = ['P9RT', 'BPN', 'P9LT']
        
        for genotype in genotypes:
            print(f"\nProcessing {genotype} genotype...")
            calculate_position_velocity_correlation(file_path, genotype, parent_dir)
            print(f"\nResults saved to {parent_dir}/{genotype}/")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data file is available in one of the expected locations.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 