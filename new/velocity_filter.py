import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_data(data_path):
    """Load data from CSV or parquet file."""
    if str(data_path).endswith('.csv'):
        return pd.read_csv(data_path)
    elif str(data_path).endswith('.parquet'):
        return pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

def filter_by_velocity(data_path="Z:/Divya/TEMP_transfers/toAni/BPN_P9LT_P9RT_flyCoords.csv"):
    """Filter trials by velocity thresholding."""
    print(f"\nFiltering trials by velocity in data from {data_path}...")
    
    # Load data
    print("Loading data...")
    df = load_data(data_path)
    
    # Print basic info
    print(f"Original data shape: {df.shape}")
    
    # Check if genotype column exists
    if 'genotype' not in df.columns:
        print("Warning: No genotype column found in data")
    else:
        # Print genotype distribution
        print("\nGenotype distribution in original data:")
        genotype_counts = df['genotype'].value_counts()
        for genotype, count in genotype_counts.items():
            print(f"  {genotype}: {count} frames")
    
    # Split data into trials (1400 frames per trial)
    original_trial_size = 1400
    num_trials = len(df) // original_trial_size
    
    print(f"Original number of trials: {num_trials}")
    
    # Track which trials pass and fail the threshold
    passing_trials = []
    failing_trials = []
    
    # Process each trial using the exact logic from train_lstm.py
    for i in tqdm(range(num_trials), desc="Filtering trials"):
        trial_start = i * original_trial_size
        trial_end = (i + 1) * original_trial_size
        trial_data = df.iloc[trial_start:trial_end]
        
        # Get genotype for this trial
        genotype = trial_data['genotype'].iloc[0]
        
        # Apply different velocity thresholds based on genotype
        passes_threshold = False
        avg_vel = 0
        vel_type = ""
        
        if genotype in ['BPN', 'P9RT', 'P9LT']:
            # For regular genotypes, check velocity in frames 350-1000
            frames = trial_data.iloc[350:1000]
            
            # Apply velocity threshold
            if genotype == 'BPN':
                # BPN: Check if average x-velocity exceeds 5 mm/s
                avg_vel = frames['x_vel'].abs().mean()
                vel_type = "x"
                passes_threshold = avg_vel >= 5
            else:
                # P9RT/P9LT: Check if average z-velocity exceeds 3 mm/s
                avg_vel = frames['z_vel'].abs().mean()
                vel_type = "z"
                passes_threshold = avg_vel >= 3
    
        # Track this trial
        if passes_threshold:
            passing_trials.append((i, genotype, avg_vel, vel_type))
        else:
            failing_trials.append((i, genotype, avg_vel, vel_type))
    
    # Print results
    print(f"\nAfter velocity thresholding:")
    print(f"Passing trials: {len(passing_trials)} (out of {num_trials})")
    print(f"Failing trials: {len(failing_trials)} (out of {num_trials})")
    
    # Print genotype distribution in passing trials
    if passing_trials:
        print("\nGenotype distribution in passing trials:")
        genotype_counts = {}
        for _, genotype, _, _ in passing_trials:
            genotype_counts[genotype] = genotype_counts.get(genotype, 0) + 1
        
        for genotype, count in genotype_counts.items():
            print(f"  {genotype}: {count} trials")
    
    # Print details of failing trials
    if failing_trials:
        print("\nFailing trials:")
        for i, (trial_idx, genotype, avg_vel, vel_type) in enumerate(failing_trials[:20]):  # Show first 20
            print(f"  Trial {trial_idx}: {genotype}, avg {vel_type}-velocity: {avg_vel:.2f} mm/s")
        
        if len(failing_trials) > 20:
            print(f"  ... and {len(failing_trials) - 20} more")
    
    # Create a visualization of passing vs failing trials
    plt.figure(figsize=(12, 8))
    
    # Group by genotype
    genotypes = set(g for _, g, _, _ in passing_trials + failing_trials)
    
    # Create subplots for each genotype
    for i, genotype in enumerate(sorted(genotypes)):
        plt.subplot(len(genotypes), 1, i+1)
        
        # Get trials for this genotype
        pass_trials = [(idx, vel, t) for idx, g, vel, t in passing_trials if g == genotype]
        fail_trials = [(idx, vel, t) for idx, g, vel, t in failing_trials if g == genotype]
        
        # Plot passing trials
        if pass_trials:
            indices, velocities, types = zip(*pass_trials)
            plt.scatter(indices, velocities, c='green', alpha=0.7, label=f'Passing ({len(pass_trials)})')
        
        # Plot failing trials
        if fail_trials:
            indices, velocities, types = zip(*fail_trials)
            plt.scatter(indices, velocities, c='red', alpha=0.7, label=f'Failing ({len(fail_trials)})')
        
        # Add threshold line
        if genotype == 'BPN':
            plt.axhline(y=5, color='black', linestyle='--', label='Threshold (5 mm/s)')
        else:
            plt.axhline(y=3, color='black', linestyle='--', label='Threshold (3 mm/s)')
        
        plt.title(f'Genotype: {genotype}')
        plt.ylabel('Average Velocity (mm/s)')
        plt.legend()
        
        if i == len(genotypes) - 1:
            plt.xlabel('Trial Index')
    
    plt.tight_layout()
    plt.savefig('velocity_filtering_results.png')
    print("\nVisualization saved to velocity_filtering_results.png")
    
    # Return the lists of passing and failing trials
    return passing_trials, failing_trials

if __name__ == "__main__":
    # Use the fixed data path
    data_path = "Z:/Divya/TEMP_transfers/toAni/BPN_P9LT_P9RT_flyCoords.csv"
    
    # Filter the trials
    passing_trials, failing_trials = filter_by_velocity(data_path) 