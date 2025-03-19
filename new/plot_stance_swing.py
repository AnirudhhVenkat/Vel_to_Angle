import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os
import re
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

def load_predictions(npz_file):
    """Load predictions from trial_predictions.npz format."""
    try:
        data = np.load(npz_file, allow_pickle=True)
        
        # Extract data from npz file
        predictions = data['predictions'].item()  # Convert from numpy object to dictionary
        targets = data['targets'].item()
        output_features = data['output_features']
        frames_per_trial = int(data['frames_per_trial'])
        sequence_length = int(data['sequence_length'])
        test_trial_metadata = data['test_trial_metadata'].item() if 'test_trial_metadata' in data else {}
        
        return predictions, targets, output_features, frames_per_trial, sequence_length, test_trial_metadata
    except Exception as e:
        print(f"Error loading {npz_file}: {e}")
        return None, None, None, None, None, None

def detect_phases(angle_data, threshold=150, min_phase_len=10):
    """
    Detect stance and swing phases based on joint angle thresholds.
    
    Args:
        angle_data (np.ndarray): Array of joint angles
        threshold (float): Angle threshold for stance/swing transition (degrees)
        min_phase_len (int): Minimum length of a phase in frames
        
    Returns:
        tuple: Arrays of stance and swing phase indices, and lists of phase start/end frames
    """
    # Handle NaN values by creating a copy and replacing NaNs
    valid_mask = ~np.isnan(angle_data)
    if not np.all(valid_mask):
        print(f"Warning: Found {np.sum(~valid_mask)} NaN values in angle data")
        # Only use valid portions
        angle_data = angle_data[valid_mask]
    
    if len(angle_data) < min_phase_len*2:
        print(f"Warning: Angle data too short ({len(angle_data)} frames) for reliable phase detection")
        return np.array([]), np.array([]), [], []
    
    # Mark frames as stance (True) when angle exceeds threshold, else swing (False)
    stance_mask = angle_data > threshold
    
    # Find transitions (0->1 for stance start, 1->0 for stance end)
    transitions = np.diff(stance_mask.astype(int))
    stance_start_indices = np.where(transitions == 1)[0] + 1
    stance_end_indices = np.where(transitions == -1)[0] + 1
    
    # Handle boundary conditions
    if stance_mask[0]:
        # First frame is stance, so prepend a start index of 0
        stance_start_indices = np.insert(stance_start_indices, 0, 0)
    
    if stance_mask[-1]:
        # Last frame is stance, so append an end index at the end
        stance_end_indices = np.append(stance_end_indices, len(angle_data))
    
    # Lists to store valid phase start/end frames
    valid_stance_starts = []
    valid_stance_ends = []
    valid_swing_starts = []
    valid_swing_ends = []
    
    # Create arrays for stance and swing indices
    stance_indices = []
    swing_indices = []
    
    # Process stance phases
    for i in range(min(len(stance_start_indices), len(stance_end_indices))):
        start = stance_start_indices[i]
        end = stance_end_indices[i]
        
        # Check if this phase is long enough
        if end - start >= min_phase_len:
            stance_indices.extend(range(start, end))
            valid_stance_starts.append(start)
            valid_stance_ends.append(end)
    
    # Process swing phases
    # First swing phase (if data doesn't start with stance)
    if len(stance_start_indices) > 0 and stance_start_indices[0] > 0:
        start = 0
        end = stance_start_indices[0]
        if end - start >= min_phase_len:
            swing_indices.extend(range(start, end))
            valid_swing_starts.append(start)
            valid_swing_ends.append(end)
    
    # Middle swing phases
    for i in range(min(len(stance_end_indices), len(stance_start_indices) - 1)):
        start = stance_end_indices[i]
        end = stance_start_indices[i + 1]
        
        # Check if this phase is long enough
        if end - start >= min_phase_len:
            swing_indices.extend(range(start, end))
            valid_swing_starts.append(start)
            valid_swing_ends.append(end)
    
    # Last swing phase (if data doesn't end with stance)
    if len(stance_end_indices) > 0 and stance_end_indices[-1] < len(angle_data):
        start = stance_end_indices[-1]
        end = len(angle_data)
        if end - start >= min_phase_len:
            swing_indices.extend(range(start, end))
            valid_swing_starts.append(start)
            valid_swing_ends.append(end)
    
    # Convert to numpy arrays
    stance_phases = list(zip(valid_stance_starts, valid_stance_ends)) if valid_stance_starts else []
    swing_phases = list(zip(valid_swing_starts, valid_swing_ends)) if valid_swing_starts else []
    
    return np.array(stance_indices), np.array(swing_indices), stance_phases, swing_phases

def calculate_phase_durations(stance_phases, swing_phases):
    """
    Calculate durations of stance and swing phases.
    
    Args:
        stance_phases (list): List of (start, end) tuples for stance phases
        swing_phases (list): List of (start, end) tuples for swing phases
        
    Returns:
        tuple: Lists of stance and swing phase durations
    """
    stance_durations = [end - start for start, end in stance_phases]
    swing_durations = [end - start for start, end in swing_phases]
    
    return stance_durations, swing_durations

def find_feti_joint(output_features):
    """Find the FeTi joint in the output features."""
    # Look for any joint with 'Fe' and 'Ti' in the name, typically B_flex or C_flex
    feti_idx = None
    feti_name = None
    
    for i, feature in enumerate(output_features):
        if ('Fe' in feature and 'Ti' in feature) or ('fe' in feature and 'ti' in feature) or 'flex' in feature.lower():
            feti_idx = i
            feti_name = feature
            break
    
    return feti_idx, feti_name

def process_predictions(results_dir):
    """Find and process all prediction files to extract phase data."""
    results_dir = Path(results_dir)
    
    # Initialize data structure for each genotype
    data_by_genotype = {
        'BPN': {'legs': [], 'stance_durations': [], 'swing_durations': [], 'stance_counts': [], 'swing_counts': []},
        'P9RT': {'legs': [], 'stance_durations': [], 'swing_durations': [], 'stance_counts': [], 'swing_counts': []},
        'P9LT': {'legs': [], 'stance_durations': [], 'swing_durations': [], 'stance_counts': [], 'swing_counts': []},
        'ES': {'legs': [], 'stance_durations': [], 'swing_durations': [], 'stance_counts': [], 'swing_counts': []}
    }
    
    # Check if directory exists
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist!")
        return data_by_genotype
    
    # Find all leg directories
    leg_dirs = [d for d in results_dir.iterdir() if d.is_dir() and re.match(r'[RL][123]', d.name)]
    
    if not leg_dirs:
        print(f"No leg directories found in {results_dir}")
        return data_by_genotype
    
    # Process each leg directory
    for leg_dir in leg_dirs:
        leg_name = leg_dir.name
        print(f"\nProcessing {leg_name}...")
        
        # Find prediction file
        pred_locations = [
                leg_dir / 'plots' / 'predictions' / 'trial_predictions.npz',
            leg_dir / 'plots' / 'trial_predictions.npz',
                leg_dir / 'predictions' / 'trial_predictions.npz',
            leg_dir / 'trial_predictions.npz',
            results_dir / 'predictions' / f'{leg_name}_trial_predictions.npz',
            results_dir / f'{leg_name}_trial_predictions.npz'
        ]
        
        pred_file = None
        for loc in pred_locations:
            if loc.exists():
                pred_file = loc
            break
    
        if not pred_file:
            print(f"Could not find predictions for {leg_name}, skipping...")
            continue
        
        print(f"Found predictions at {pred_file}")
    
    # Load predictions
        predictions, targets, output_features, frames_per_trial, sequence_length, test_trial_metadata = load_predictions(pred_file)
        
        if predictions is None:
            print(f"Failed to load predictions for {leg_name}, skipping...")
                continue
                
        # Find FeTi joint (for phase detection)
        feti_idx, feti_name = find_feti_joint(output_features)
            
        if feti_idx is None:
            print(f"Could not find FeTi joint for {leg_name}, skipping...")
                continue
                
        print(f"Using joint {feti_name} for phase detection")
        
        # Create genotype-specific collections for this leg
        genotype_stance_durations = {'BPN': [], 'P9RT': [], 'P9LT': [], 'ES': []}
        genotype_swing_durations = {'BPN': [], 'P9RT': [], 'P9LT': [], 'ES': []}
        genotype_stance_counts = {'BPN': [], 'P9RT': [], 'P9LT': [], 'ES': []}
        genotype_swing_counts = {'BPN': [], 'P9RT': [], 'P9LT': [], 'ES': []}
        
        # Process each trial
        for trial_idx in sorted(predictions.keys()):
            # Get genotype for this trial
            if trial_idx in test_trial_metadata:
                genotype = test_trial_metadata[trial_idx].get('genotype', "Unknown")
            else:
                print(f"No genotype information for trial {trial_idx}, skipping...")
                continue
                
            # Skip trials with unknown genotype
            if genotype not in ['BPN', 'P9RT', 'P9LT', 'ES']:
                print(f"Unknown genotype {genotype} for trial {trial_idx}, skipping...")
                    continue
                
            # Get target data for FeTi joint
            if feti_name not in targets[trial_idx]:
                print(f"Joint {feti_name} not found in trial {trial_idx}, skipping...")
                    continue
                
            target_angles = targets[trial_idx][feti_name]
            
            # Detect phases with improved function
            stance_indices, swing_indices, stance_phases, swing_phases = detect_phases(target_angles)
            
            # Calculate phase durations with improved function
            stance_durations, swing_durations = calculate_phase_durations(stance_phases, swing_phases)
            
            # Count number of complete phases per trial
            genotype_stance_counts[genotype].append(len(stance_phases))
            genotype_swing_counts[genotype].append(len(swing_phases))
            
            # Add durations to collections
            genotype_stance_durations[genotype].extend(stance_durations)
            genotype_swing_durations[genotype].extend(swing_durations)
        
        # Add data to the main data structure
        for genotype in ['BPN', 'P9RT', 'P9LT', 'ES']:
            if genotype_stance_durations[genotype] or genotype_swing_durations[genotype]:
                if leg_name not in data_by_genotype[genotype]['legs']:
                    data_by_genotype[genotype]['legs'].append(leg_name)
                    data_by_genotype[genotype]['stance_durations'].append([])
                    data_by_genotype[genotype]['swing_durations'].append([])
                    data_by_genotype[genotype]['stance_counts'].append([])
                    data_by_genotype[genotype]['swing_counts'].append([])
                
                leg_idx = data_by_genotype[genotype]['legs'].index(leg_name)
                data_by_genotype[genotype]['stance_durations'][leg_idx].extend(genotype_stance_durations[genotype])
                data_by_genotype[genotype]['swing_durations'][leg_idx].extend(genotype_swing_durations[genotype])
                data_by_genotype[genotype]['stance_counts'][leg_idx].extend(genotype_stance_counts[genotype])
                data_by_genotype[genotype]['swing_counts'][leg_idx].extend(genotype_swing_counts[genotype])
        
        print(f"Processed {leg_name} by genotype:")
        for genotype in ['BPN', 'P9RT', 'P9LT', 'ES']:
            print(f"  {genotype}: {len(genotype_stance_durations[genotype])} stance phases, {len(genotype_swing_durations[genotype])} swing phases")
    
    # Print summary
    for genotype in data_by_genotype:
        legs = data_by_genotype[genotype]['legs']
        if legs:
            print(f"\n{genotype} data:")
            for i, leg in enumerate(legs):
                stance = data_by_genotype[genotype]['stance_durations'][i]
                swing = data_by_genotype[genotype]['swing_durations'][i]
                print(f"  {leg}: {len(stance)} stance phases, {len(swing)} swing phases")
    
    return data_by_genotype

def create_boxplots_by_genotype(data_by_genotype, output_dir):
    """Create separate box plots for each genotype, showing all 6 legs."""
    genotypes = ['BPN', 'P9RT', 'P9LT', 'ES']
    all_legs = ['R1', 'L1', 'R2', 'L2', 'R3', 'L3']  # All legs should be shown
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Process each genotype
    for genotype in genotypes:
        if genotype not in data_by_genotype:
            print(f"No data available for genotype {genotype}, skipping...")
                continue
            
        genotype_data = data_by_genotype[genotype]
        
        # Create figure for this genotype
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        fig.suptitle(f'{genotype} - Stance and Swing Phase Durations', fontsize=16)
        
        # Prepare data for all legs, including empty ones
        stance_data = []
        swing_data = []
        
        # Initialize with empty lists for all legs
        for leg in all_legs:
            if leg in genotype_data['legs']:
                # If we have data for this leg, use it
                leg_idx = genotype_data['legs'].index(leg)
                stance_data.append(genotype_data['stance_durations'][leg_idx])
                swing_data.append(genotype_data['swing_durations'][leg_idx])
            else:
                # No data for this leg, use empty list
                stance_data.append([])
                swing_data.append([])
        
        # Create boxplots for stance phases
        sns.boxplot(data=stance_data, ax=ax1)
        ax1.set_xticklabels(all_legs)
        ax1.set_title(f'{genotype} - Stance Phase Durations')
        ax1.set_ylabel('Duration (frames)')
        ax1.set_xlabel('Leg')
        
        # Show sample sizes
        for i, leg in enumerate(all_legs):
            n_samples = len(stance_data[i])
            if n_samples > 0:
                ax1.annotate(f'n={n_samples}', 
                          xy=(i, 5), 
                          ha='center',
                          fontsize=9)
        
        # Create boxplots for swing phases
        sns.boxplot(data=swing_data, ax=ax2)
        ax2.set_xticklabels(all_legs)
        ax2.set_title(f'{genotype} - Swing Phase Durations')
        ax2.set_ylabel('Duration (frames)')
        ax2.set_xlabel('Leg')
        
        # Show sample sizes
        for i, leg in enumerate(all_legs):
            n_samples = len(swing_data[i])
            if n_samples > 0:
                ax2.annotate(f'n={n_samples}', 
                          xy=(i, 5), 
                          ha='center',
                          fontsize=9)
        
        # Save figure
        plt.tight_layout()
        output_file = output_dir / f'{genotype}_phase_durations.pdf'
        plt.savefig(output_file)
                    plt.close()
            
        print(f"Created box plots for {genotype} showing all 6 legs at {output_file}")
        
        # Create a third figure for phase count statistics
        fig_counts, (ax_stance, ax_swing) = plt.subplots(1, 2, figsize=(14, 8))
        fig_counts.suptitle(f'{genotype} - Number of Stance/Swing Phases per Trial', fontsize=16)
        
        # Prepare data for all legs, including empty ones
        stance_counts = []
        swing_counts = []
        
        # Initialize with empty lists for all legs
        for leg in all_legs:
            if leg in genotype_data['legs']:
                # If we have data for this leg, use it
                leg_idx = genotype_data['legs'].index(leg)
                stance_counts.append(genotype_data['stance_counts'][leg_idx])
                swing_counts.append(genotype_data['swing_counts'][leg_idx])
            else:
                # No data for this leg, use empty list
                stance_counts.append([])
                swing_counts.append([])
        
        # Calculate average counts for each leg
        avg_stance_counts = [np.mean(counts) if counts else 0 for counts in stance_counts]
        avg_swing_counts = [np.mean(counts) if counts else 0 for counts in swing_counts]
        
        # Create bar plots for stance phases
        ax_stance.bar(range(len(all_legs)), avg_stance_counts)
        ax_stance.set_xticks(range(len(all_legs)))
        ax_stance.set_xticklabels(all_legs)
        ax_stance.set_title(f'{genotype} - Average Stance Phases per Trial')
        ax_stance.set_ylabel('Count')
        ax_stance.set_xlabel('Leg')
        
        # Show sample sizes
        for i, leg in enumerate(all_legs):
            n_trials = len(stance_counts[i])
            if n_trials > 0:
                ax_stance.annotate(f'n={n_trials} trials', 
                          xy=(i, avg_stance_counts[i] + 0.2), 
                          ha='center',
                          fontsize=9)
        
        # Create bar plots for swing phases
        ax_swing.bar(range(len(all_legs)), avg_swing_counts)
        ax_swing.set_xticks(range(len(all_legs)))
        ax_swing.set_xticklabels(all_legs)
        ax_swing.set_title(f'{genotype} - Average Swing Phases per Trial')
        ax_swing.set_ylabel('Count')
        ax_swing.set_xlabel('Leg')
        
        # Show sample sizes
        for i, leg in enumerate(all_legs):
            n_trials = len(swing_counts[i])
            if n_trials > 0:
                ax_swing.annotate(f'n={n_trials} trials', 
                          xy=(i, avg_swing_counts[i] + 0.2), 
                          ha='center',
                          fontsize=9)
        
        # Save figure
        plt.tight_layout()
        output_file = output_dir / f'{genotype}_phase_counts.pdf'
        plt.savefig(output_file)
        plt.close()
        
        print(f"Created phase count plots for {genotype} at {output_file}")

def main():
    """Main function to create genotype-specific box plots."""
    # Define directories
    results_dir = Path('lstm_results')
    output_dir = Path('stance_swing_analysis/box_plots')
    
    # Print paths for clarity
    print(f"Looking for results in: {results_dir.absolute()}")
    print(f"Output will be saved to: {output_dir.absolute()}")
    
    # Check if directory exists
    if not results_dir.exists():
        print(f"Results directory {results_dir} not found!")
        print("Current working directory:", Path.cwd())
        user_dir = input("Enter the full path to the lstm_results directory (or press Enter to abort): ")
        
        if not user_dir:
            print("Aborting...")
            return
        
        results_dir = Path(user_dir)
        if not results_dir.exists():
            print(f"Directory {results_dir} still not found. Aborting...")
            return
    
    # Process predictions to extract phase data
    data_by_genotype = process_predictions(results_dir)
    
    # Create box plots for each genotype
    create_boxplots_by_genotype(data_by_genotype, output_dir)
    
    print("\nBox plot generation completed!")

if __name__ == "__main__":
    main() 