import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

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

def load_and_filter_data(file_path, genotype):
    """Load and preprocess the dataset from a CSV file."""
    try:
        # Load the full data
        df = pd.read_csv(file_path)
        print(f"Raw data loaded successfully with shape: {df.shape}")
        
        # Filter for genotype
        if genotype:
            df = df[df['genotype'] == genotype]
            print(f"Filtered data for {genotype} genotype, new shape: {df.shape}")    
        
        # Filter frames 400-1000 from each trial
        trial_size = 1400
        num_trials = len(df) // trial_size
        
        filtered_rows = []
        for trial in range(num_trials):
            start_idx = trial * trial_size + 400
            end_idx = trial * trial_size + 1000
            filtered_rows.append(df.iloc[start_idx:end_idx])
        
        df = pd.concat(filtered_rows, axis=0, ignore_index=True)
        print(f"Filtered data to frames 400-1000, new shape: {df.shape}")
        
        return df
        
    except Exception as e:
        print(f"Error loading/preprocessing data: {e}")
        raise

def calculate_coordinate_joint_correlations(data_path, genotype, output_dir='coordinate_joint_correlations'):
    """Calculate Spearman correlations between coordinates and joint angles."""
    # Create output directory
    output_path = Path(output_dir) / genotype
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    print(f"\nLoading and preprocessing data for {genotype}...")
    df = load_and_filter_data(data_path, genotype)
    
    # Get coordinate columns (ending in _x, _y, _z)
    coord_cols = [col for col in df.columns if col.endswith(('_x', '_y', '_z'))]
    
    # Get joint angle columns
    joint_cols = [col for col in df.columns if col.endswith(('_flex', '_rot', '_abduct'))]
    
    print(f"\nAnalyzing correlations between {len(coord_cols)} coordinates and {len(joint_cols)} joint angles")
    
    # Initialize results list
    correlations = []
    
    # Calculate correlations
    print("\nCalculating Spearman correlations...")
    for coord in coord_cols:
        for joint in joint_cols:
            try:
                # Get data as numpy arrays
                x = df[coord].to_numpy().ravel()
                y = df[joint].to_numpy().ravel()
                
                # Remove NaN values
                mask = ~(np.isnan(x) | np.isnan(y))
                x = x[mask]
                y = y[mask]
                
                # Calculate correlation
                if len(x) > 0 and len(y) > 0:
                    correlation, p_value = stats.spearmanr(x, y)
                    
                    # Store significant correlations (p < 0.05 and |correlation| > 0.3)
                    if abs(correlation) > 0.3 and p_value < 0.05:
                        correlations.append({
                            'Coordinate': coord,
                            'Joint Angle': joint,
                            'Correlation': correlation,
                            'P-value': p_value
                        })
                        
            except Exception as e:
                print(f"\nError calculating correlation between {coord} and {joint}: {e}")
                continue
    
    # Convert to DataFrame and sort by absolute correlation
    if correlations:
        corr_df = pd.DataFrame(correlations)
        corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
        corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
        corr_df = corr_df.drop('Abs_Correlation', axis=1)
        
        # Save to CSV
        csv_path = output_path / f'coordinate_joint_correlations_{genotype}.csv'
        corr_df.to_csv(csv_path, index=False)
        print(f"\nSaved correlations to: {csv_path}")
        
        # Create correlation matrix for heatmap
        pivot_df = corr_df.pivot(index='Coordinate', 
                                columns='Joint Angle', 
                                values='Correlation')
        
        # Plot heatmap
        plt.figure(figsize=(20, 10))
        sns.heatmap(pivot_df, 
                   annot=True, 
                   cmap='coolwarm',
                   center=0,
                   fmt='.2f',
                   vmin=-1,
                   vmax=1)
        
        plt.title(f'Coordinate-Joint Angle Correlations ({genotype})')
        plt.tight_layout()
        
        # Save heatmap
        heatmap_path = output_path / f'coordinate_joint_correlations_heatmap_{genotype}.png'
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print top correlations
        print(f"\nTop 20 strongest correlations for {genotype}:")
        print(corr_df.head(20).to_string(index=False))
        
        return corr_df
    else:
        print(f"\nNo significant correlations found for {genotype}")
        return None

def main():
    try:
        # Get data path
        file_path = get_available_data_path()
        
        # Process each genotype
        genotypes = ['P9RT', 'BPN', 'P9LT']
        
        for genotype in genotypes:
            print(f"\nProcessing {genotype} genotype...")
            correlations = calculate_coordinate_joint_correlations(file_path, genotype)
            
            if correlations is not None:
                print(f"\nFound {len(correlations)} significant correlations for {genotype}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 