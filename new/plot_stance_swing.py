import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
import seaborn as sns
from matplotlib.patches import Rectangle
import json
from matplotlib.backends.backend_pdf import PdfPages
import torch
import warnings
import pandas as pd

class ZScoreScaler:
    """Z-score normalization scaler"""
    def __init__(self, means=None, stds=None, feature_names=None):
        self.means = means
        self.stds = stds
        self.feature_names = feature_names
    
    def fit(self, X):
        """Fit the scaler to the data."""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        # Handle constant features
        self.stds[self.stds == 0] = 1
        return self
    
    def transform(self, X):
        """Transform the data."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return (X - self.means) / self.stds
    
    def fit_transform(self, X):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """Convert back to original scale."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return X * self.stds + self.means

def load_model_checkpoint(model_path):
    """Load model checkpoint with proper error handling."""
    try:
        # First try loading with weights_only=True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            checkpoint = torch.load(model_path, weights_only=True)
        return checkpoint
    except Exception as e:
        print(f"Warning: Could not load with weights_only=True, trying full load: {e}")
        try:
            # Try full load if weights_only fails
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                checkpoint = torch.load(model_path)
            return checkpoint
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None

def load_predictions(npz_path):
    """Load predictions from NPZ file."""
    data = np.load(npz_path)
    
    # Load predictions and targets
    predictions = data['predictions']
    targets = data['targets']
    output_features = data['output_features']
    frames = data.get('frames', np.arange(predictions.shape[1]))  # Default to sequential frames if not provided
    
    # Get number of trials
    num_trials = predictions.shape[0]
    
    print(f"Loaded {num_trials} trials")
    
    return predictions, targets, output_features, frames

def detect_phases(angle_data, prominence=5):
    """
    Detect stance and swing phases based on peaks in angle data.
    Stance phase: Leading up to peak
    Swing phase: Going down from peak
    """
    # Find peaks with minimum prominence to avoid detecting small fluctuations
    peaks, _ = find_peaks(angle_data, prominence=prominence)
    
    # Initialize phase arrays
    phases = np.zeros_like(angle_data, dtype=int)  # 0: unknown, 1: stance, 2: swing
    
    if len(peaks) < 2:
        return phases, peaks
    
    # Process each peak
    for i in range(len(peaks)-1):
        current_peak = peaks[i]
        next_peak = peaks[i+1]
        
        # Find minimum between peaks
        valley_idx = current_peak + np.argmin(angle_data[current_peak:next_peak])
        
        # Mark stance phase (rising to peak)
        phases[current_peak-10:current_peak+1] = 1  # Include some frames before peak
        
        # Mark swing phase (falling from peak)
        phases[current_peak+1:valley_idx+1] = 2
    
    # Handle last peak
    if len(peaks) > 0:
        last_peak = peaks[-1]
        phases[last_peak-10:last_peak+1] = 1
        phases[last_peak+1:last_peak+20] = 2  # Assume swing for some frames after
    
    return phases, peaks

def calculate_phase_durations(phases):
    """Calculate durations of stance and swing phases."""
    stance_durations = []
    swing_durations = []
    
    # Find continuous regions of each phase
    for phase_type in [1, 2]:  # 1: stance, 2: swing
        phase_regions = np.where(phases == phase_type)[0]
        if len(phase_regions) > 0:
            # Split into continuous segments
            splits = np.where(np.diff(phase_regions) > 1)[0] + 1
            segments = np.split(phase_regions, splits)
            
            # Calculate durations
            durations = [len(seg) for seg in segments if len(seg) > 0]
            
            if phase_type == 1:
                stance_durations.extend(durations)
            else:
                swing_durations.extend(durations)
    
    return stance_durations, swing_durations

def plot_phase_durations(actual_stance, actual_swing, pred_stance, pred_swing, leg_names, save_path):
    """Create box plots comparing actual vs predicted stance and swing phase durations."""
    plt.figure(figsize=(15, 8))
    plt.suptitle('Distribution of Stance and Swing Phase Durations (B_flex angles)', fontsize=16, y=0.95)
    
    # Create two subplots side by side
    # Stance phases
    plt.subplot(1, 2, 1)
    bp = plt.boxplot([durations for durations in actual_stance if len(durations) > 0],
                     positions=np.arange(len(leg_names)) * 2,
                     labels=[f"{leg}\nActual" for leg in leg_names],
                     patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    bp = plt.boxplot([durations for durations in pred_stance if len(durations) > 0],
                     positions=np.arange(len(leg_names)) * 2 + 1,
                     labels=[f"{leg}\nPred" for leg in leg_names],
                     patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
    
    plt.title('Stance Phase Durations')
    plt.ylabel('Duration (frames)')
    plt.xticks(rotation=45, ha='right')
    
    # Swing phases
    plt.subplot(1, 2, 2)
    bp = plt.boxplot([durations for durations in actual_swing if len(durations) > 0],
                     positions=np.arange(len(leg_names)) * 2,
                     labels=[f"{leg}\nActual" for leg in leg_names],
                     patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    bp = plt.boxplot([durations for durations in pred_swing if len(durations) > 0],
                     positions=np.arange(len(leg_names)) * 2 + 1,
                     labels=[f"{leg}\nPred" for leg in leg_names],
                     patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
    
    plt.title('Swing Phase Durations')
    plt.ylabel('Duration (frames)')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_trial_phases(predictions, targets, frames, trial_idx, leg_name, pdf):
    """Plot stance and swing phases for B_flex angle of a single trial."""
    fig = plt.figure(figsize=(12, 6))
    plt.suptitle(f'{leg_name} - Trial {trial_idx + 1} - B_flex Stance and Swing Phases', fontsize=16, y=0.95)
    
    # Create color maps for phases
    colors = {
        1: 'lightgreen',  # stance
        2: 'lightblue'    # swing
    }
    
    # Get predictions and targets
    pred = predictions
    target = targets
    
    # Detect phases for actual data
    phases, peaks = detect_phases(target)
    
    # Plot phases as background colors
    for phase in [1, 2]:  # 1: stance, 2: swing
        phase_regions = np.where(phases == phase)[0]
        if len(phase_regions) > 0:
            for start_idx in np.where(np.diff(np.hstack(([0], phase_regions))) > 1)[0]:
                end_idx = start_idx
                while end_idx < len(phase_regions) and phase_regions[end_idx] - phase_regions[start_idx] <= end_idx - start_idx + 1:
                    end_idx += 1
                rect = Rectangle((frames[phase_regions[start_idx]], plt.ylim()[0]),
                               frames[phase_regions[end_idx-1]] - frames[phase_regions[start_idx]],
                               plt.ylim()[1] - plt.ylim()[0],
                               facecolor=colors[phase], alpha=0.3)
                plt.gca().add_patch(rect)
    
    # Plot predictions and targets
    plt.plot(frames, target, 'b-', label='Actual', linewidth=2)
    plt.plot(frames, pred, 'r--', label='Predicted', linewidth=2, alpha=0.7)
    
    # Plot peaks
    plt.plot(frames[peaks], target[peaks], 'k^', label='Peaks')
    
    plt.title(f'{leg_name}B_flex')
    plt.xlabel('Frame')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)
    plt.close()

def get_trial_genotypes(data_path, trial_indices=None):
    """Load the original data and get genotype for each trial."""
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded genotype data with shape: {df.shape}")
    print(f"Available genotypes: {df['genotype'].unique()}")
    
    # Get trial indices and genotypes
    trial_size = 1400
    num_trials = len(df) // trial_size
    print(f"Number of trials in genotype data: {num_trials}")
    
    # Get genotype for each trial (using first frame of trial)
    trial_genotypes = {}
    for trial in range(num_trials):
        start_idx = trial * trial_size
        genotype = df.iloc[start_idx]['genotype']
        trial_genotypes[trial] = genotype
    
    # If trial_indices is provided, reorder genotypes according to the permutation
    if trial_indices is not None:
        reordered_genotypes = {}
        for new_idx, old_idx in enumerate(trial_indices):
            reordered_genotypes[new_idx] = trial_genotypes[old_idx]
        trial_genotypes = reordered_genotypes
    
    # Print genotype distribution
    genotype_counts = {}
    for genotype in trial_genotypes.values():
        genotype_counts[genotype] = genotype_counts.get(genotype, 0) + 1
    print("\nGenotype distribution:")
    for genotype, count in genotype_counts.items():
        print(f"{genotype}: {count} trials")
    
    return trial_genotypes

def process_leg_predictions(leg_name, results_dir='lstm_results'):
    """Process predictions for B_flex angle of a specific leg."""
    print(f"\nProcessing {leg_name}B_flex predictions...")
    
    # Find the leg directory
    leg_dir = Path(results_dir) / leg_name
    if not leg_dir.exists():
        print(f"No results directory found for {leg_name}")
        return None
    
    # Look for predictions file
    predictions_path = leg_dir / 'plots' / 'predictions' / 'trial_predictions.npz'
    if not predictions_path.exists():
        print(f"No predictions found for {leg_name}")
        return None
    
    print(f"Found predictions at: {predictions_path}")
    
    # Create output directory
    output_dir = leg_dir / 'stance_swing_analysis'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # Load original data first to get trial indices
        data_path = "Z:/Divya/TEMP_transfers/toAni/BPN_P9LT_P9RT_flyCoords.csv"
        df = pd.read_csv(data_path)
        print(f"Loaded original data with shape: {df.shape}")
        
        # Get trial indices using same logic as train_lstm.py
        trial_size = 1400
        num_trials = len(df) // trial_size
        print(f"Total number of trials in original data: {num_trials}")
        
        # Create random permutation of trial indices (same as train_lstm.py)
        np.random.seed(42)  # Same seed as train_lstm.py
        trial_indices = np.random.permutation(num_trials)
        
        # Calculate split sizes (same as train_lstm.py)
        train_size = int(0.7 * num_trials)
        val_size = int(0.15 * num_trials)
        
        # Split trial indices
        train_trials = trial_indices[:train_size]
        val_trials = trial_indices[train_size:train_size + val_size]
        test_trials = trial_indices[train_size + val_size:]
        
        print("\nTrial split information:")
        print(f"Training trials: {len(train_trials)}")
        print(f"Validation trials: {len(val_trials)}")
        print(f"Test trials: {len(test_trials)}")
        
        # Get genotype for each trial using full 1400 frames
        trial_genotypes = {}
        for trial in range(num_trials):
            start_idx = trial * trial_size
            end_idx = (trial + 1) * trial_size
            trial_data = df.iloc[start_idx:end_idx]
            genotype_counts = trial_data['genotype'].value_counts()
            majority_genotype = genotype_counts.index[0]
            
            if len(genotype_counts) > 1:
                print(f"Warning: Trial {trial} has multiple genotypes: {dict(genotype_counts)}")
            
            trial_genotypes[trial] = majority_genotype
        
        # Load predictions
        predictions, targets, output_features, frames = load_predictions(predictions_path)
        
        # Find B_flex index
        b_flex_idx = None
        for i, feature in enumerate(output_features):
            if feature.endswith('B_flex'):
                b_flex_idx = i
                break
        
        if b_flex_idx is None:
            print(f"No B_flex angle found for {leg_name}")
            return None
        
        # Initialize dictionaries to store phase durations by genotype
        genotype_data = {
            'P9RT': {'actual_stance': [], 'actual_swing': [], 'pred_stance': [], 'pred_swing': [], 'trials': [], 'pred_indices': []},
            'P9LT': {'actual_stance': [], 'actual_swing': [], 'pred_stance': [], 'pred_swing': [], 'trials': [], 'pred_indices': []},
            'BPN': {'actual_stance': [], 'actual_swing': [], 'pred_stance': [], 'pred_swing': [], 'trials': [], 'pred_indices': []},
            'ES': {'actual_stance': [], 'actual_swing': [], 'pred_stance': [], 'pred_swing': [], 'trials': [], 'pred_indices': []}
        }
        
        # Process test trials only (since predictions are for test set)
        print("\nProcessing test trials...")
        for i, trial_idx in enumerate(test_trials):
            # Get trial genotype
            genotype = trial_genotypes[trial_idx]
            if genotype not in genotype_data:
                print(f"Unknown genotype {genotype} for trial {trial_idx}, skipping...")
                continue
            
            # Get B_flex predictions and targets for this trial
            pred = predictions[i, :, b_flex_idx]  # Use i since predictions are already in test order
            target = targets[i, :, b_flex_idx]
            
            # Skip if either predictions or targets contain NaN values
            if np.isnan(pred).any() or np.isnan(target).any():
                print(f"Skipping trial {trial_idx} due to NaN values")
                continue
            
            # Calculate phase durations for actual data
            actual_phases, _ = detect_phases(target)
            trial_actual_stance, trial_actual_swing = calculate_phase_durations(actual_phases)
            
            # Calculate phase durations for predicted data
            pred_phases, _ = detect_phases(pred)
            trial_pred_stance, trial_pred_swing = calculate_phase_durations(pred_phases)
            
            # Store results by genotype
            genotype_data[genotype]['actual_stance'].extend(trial_actual_stance)
            genotype_data[genotype]['actual_swing'].extend(trial_actual_swing)
            genotype_data[genotype]['pred_stance'].extend(trial_pred_stance)
            genotype_data[genotype]['pred_swing'].extend(trial_pred_swing)
            genotype_data[genotype]['trials'].append(trial_idx)
            genotype_data[genotype]['pred_indices'].append(i)  # Store the prediction index
        
        # Print final genotype distribution for test set
        print("\nGenotype distribution in test set:")
        for genotype in genotype_data:
            num_trials = len(genotype_data[genotype]['trials'])
            print(f"{genotype}: {num_trials} trials")
        
        # Filter trials based on velocity thresholds
        filtered_trials = []
        for trial in range(num_trials):
            # Different frame ranges for ES vs other genotypes
            if genotype == 'ES':
                start_idx = trial * trial_size  # Start from frame 0 for ES
                end_idx = trial * trial_size + 650  # End at frame 650 for ES
            else:
                start_idx = trial * trial_size + 350  # Start from frame 350 for others
                end_idx = trial * trial_size + 1000   # End at frame 1000 for others
            
            trial_data = df.iloc[start_idx:end_idx]
            
            # Calculate average velocity for the trial
            if genotype in ['P9RT', 'P9LT']:
                avg_vel = trial_data['z_vel'].mean()
                threshold = 3
                vel_type = 'z'
            else:  # BPN or ES
                avg_vel = trial_data['x_vel'].mean()
                threshold = 5
                vel_type = 'x'
            
            # Keep trial if it meets the threshold
            if abs(avg_vel) >= threshold:
                filtered_trials.append(trial)
        
        print(f"\nVelocity filtering criteria for {genotype}:")
        print(f"Using {vel_type}_vel with threshold {threshold}")
        print(f"Trials remaining after filtering: {len(filtered_trials)} out of {num_trials}")
        
        # Create PDF for each genotype
        for genotype, data in genotype_data.items():
            if not data['trials']:  # Skip if no trials for this genotype
                continue
                
            print(f"\nProcessing {genotype} trials for {leg_name}...")
            genotype_dir = output_dir / genotype
            genotype_dir.mkdir(exist_ok=True)
            
            # Create PDF for this genotype's trials
            pdf_path = genotype_dir / f'{leg_name}_B_flex_trials.pdf'
            with PdfPages(pdf_path) as pdf:
                # First page: Summary statistics
                plt.figure(figsize=(12, 8))
                plt.axis('off')
                plt.text(0.1, 0.95, f'{leg_name}B_flex Phase Analysis - {genotype}', fontsize=16, fontweight='bold')
                plt.text(0.1, 0.85, f'Number of trials: {len(data["trials"])}', fontsize=12)
                plt.text(0.1, 0.75, 'Color coding:', fontsize=12)
                plt.text(0.1, 0.7, '  Green: Stance phase (leading to peak)', fontsize=12)
                plt.text(0.1, 0.65, '  Blue: Swing phase (after peak)', fontsize=12)
                plt.text(0.1, 0.6, '  Black triangles: Detected peaks', fontsize=12)
                plt.text(0.1, 0.55, '  Blue line: Actual angle', fontsize=12)
                plt.text(0.1, 0.5, '  Red dashed line: Predicted angle', fontsize=12)
                pdf.savefig()
                plt.close()
                
                # Plot each trial for this genotype
                for trial_idx, pred_idx in zip(data['trials'], data['pred_indices']):
                    # Get B_flex predictions and targets using the correct prediction index
                    pred = predictions[pred_idx, :, b_flex_idx]
                    target = targets[pred_idx, :, b_flex_idx]
                    
                    # Create trial plot
                    plot_trial_phases(pred, target, frames, trial_idx, leg_name, pdf)
                
                # Add final page with phase duration statistics
                plt.figure(figsize=(12, 8))
                plt.axis('off')
                plt.text(0.1, 0.95, f'{leg_name}B_flex Phase Duration Statistics - {genotype}', fontsize=16, fontweight='bold')
                
                # Calculate statistics for this genotype
                stats = {
                    'actual_stance': {
                        'mean': float(np.mean(data['actual_stance'])),
                        'std': float(np.std(data['actual_stance'])),
                        'median': float(np.median(data['actual_stance'])),
                        'min': float(np.min(data['actual_stance'])),
                        'max': float(np.max(data['actual_stance']))
                    },
                    'actual_swing': {
                        'mean': float(np.mean(data['actual_swing'])),
                        'std': float(np.std(data['actual_swing'])),
                        'median': float(np.median(data['actual_swing'])),
                        'min': float(np.min(data['actual_swing'])),
                        'max': float(np.max(data['actual_swing']))
                    },
                    'predicted_stance': {
                        'mean': float(np.mean(data['pred_stance'])),
                        'std': float(np.std(data['pred_stance'])),
                        'median': float(np.median(data['pred_stance'])),
                        'min': float(np.min(data['pred_stance'])),
                        'max': float(np.max(data['pred_stance']))
                    },
                    'predicted_swing': {
                        'mean': float(np.mean(data['pred_swing'])),
                        'std': float(np.std(data['pred_swing'])),
                        'median': float(np.median(data['pred_swing'])),
                        'min': float(np.min(data['pred_swing'])),
                        'max': float(np.max(data['pred_swing']))
                    }
                }
                
                # Add statistics to plot
                y_pos = 0.85
                for phase in ['actual_stance', 'predicted_stance', 'actual_swing', 'predicted_swing']:
                    plt.text(0.1, y_pos, f"\n{phase.replace('_', ' ').title()}:", fontsize=14, fontweight='bold')
                    y_pos -= 0.08
                    for stat, value in stats[phase].items():
                        plt.text(0.1, y_pos, f"  {stat}: {value:.2f} frames")
                        y_pos -= 0.05
                    y_pos -= 0.05
                
                pdf.savefig()
                plt.close()
            
            # Save statistics to JSON
            with open(genotype_dir / 'phase_duration_stats.json', 'w') as f:
                json.dump({f'{leg_name}B_flex': stats}, f, indent=4)
            
            print(f"PDF and statistics saved to {genotype_dir}")
        
        # Create combined box plots comparing genotypes
        plt.figure(figsize=(15, 10))
        plt.suptitle(f'{leg_name} Phase Durations by Genotype', fontsize=16)
        
        # Stance phase comparison
        plt.subplot(2, 1, 1)
        data = []
        labels = []
        for genotype in ['P9RT', 'P9LT', 'BPN']:
            if genotype_data[genotype]['actual_stance']:
                data.append(genotype_data[genotype]['actual_stance'])
                data.append(genotype_data[genotype]['pred_stance'])
                labels.extend([f'{genotype}\nActual', f'{genotype}\nPred'])
        
        plt.boxplot(data, labels=labels)
        plt.title('Stance Phase Duration')
        plt.ylabel('Duration (frames)')
        plt.grid(True, alpha=0.3)
        
        # Swing phase comparison
        plt.subplot(2, 1, 2)
        data = []
        labels = []
        for genotype in ['P9RT', 'P9LT', 'BPN']:
            if genotype_data[genotype]['actual_swing']:
                data.append(genotype_data[genotype]['actual_swing'])
                data.append(genotype_data[genotype]['pred_swing'])
                labels.extend([f'{genotype}\nActual', f'{genotype}\nPred'])
        
        plt.boxplot(data, labels=labels)
        plt.title('Swing Phase Duration')
        plt.ylabel('Duration (frames)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{leg_name}_phase_durations_by_genotype.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return genotype_data
    
    except Exception as e:
        print(f"Error processing {leg_name} for {results_dir}: {str(e)}")
        return None

def create_genotype_summary_plot(genotype, all_leg_data, output_dir):
    """Create a summary plot for all legs of a given genotype."""
    plt.figure(figsize=(20, 10))
    plt.suptitle(f'{genotype} Phase Durations - All Legs', fontsize=16)
    
    # Stance phase comparison (top subplot)
    plt.subplot(2, 1, 1)
    data = []
    labels = []
    for leg_name, leg_data in all_leg_data.items():
        if leg_data['actual_stance']:  # Only include if we have data
            data.append(leg_data['actual_stance'])
            data.append(leg_data['pred_stance'])
            labels.extend([f'{leg_name}\nActual', f'{leg_name}\nPred'])
    
    bp = plt.boxplot(data, labels=labels, patch_artist=True)
    
    # Color boxes
    for i in range(0, len(bp['boxes']), 2):
        bp['boxes'][i].set_facecolor('lightblue')  # Actual
        bp['boxes'][i+1].set_facecolor('lightgreen')  # Predicted
    
    plt.title('Stance Phase Duration')
    plt.ylabel('Duration (frames)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    plt.plot([], [], 'lightblue', label='Actual')
    plt.plot([], [], 'lightgreen', label='Predicted')
    plt.legend()
    
    # Swing phase comparison (bottom subplot)
    plt.subplot(2, 1, 2)
    data = []
    labels = []
    for leg_name, leg_data in all_leg_data.items():
        if leg_data['actual_swing']:  # Only include if we have data
            data.append(leg_data['actual_swing'])
            data.append(leg_data['pred_swing'])
            labels.extend([f'{leg_name}\nActual', f'{leg_name}\nPred'])
    
    bp = plt.boxplot(data, labels=labels, patch_artist=True)
    
    # Color boxes
    for i in range(0, len(bp['boxes']), 2):
        bp['boxes'][i].set_facecolor('lightblue')  # Actual
        bp['boxes'][i+1].set_facecolor('lightgreen')  # Predicted
    
    plt.title('Swing Phase Duration')
    plt.ylabel('Duration (frames)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Add legend
    plt.plot([], [], 'lightblue', label='Actual')
    plt.plot([], [], 'lightgreen', label='Predicted')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{genotype}_all_legs_phase_durations.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Process B_flex predictions for all legs."""
    # Define legs to process
    legs = ['R1', 'L1', 'R2', 'L2', 'R3', 'L3']
    
    # Define model types to check
    model_types = ['lstm_results', 'transformer_results', 'tcn_results', 'hybrid_results']
    
    # Create base output directory
    base_output_dir = Path('stance_swing_analysis')
    base_output_dir.mkdir(exist_ok=True, parents=True)
    
    # Process each model type
    for model_type in model_types:
        model_dir = Path(model_type)
        if not model_dir.exists():
            print(f"\nNo results found for {model_type}")
            continue
            
        print(f"\nProcessing {model_type}...")
        
        # Create model-specific output directory
        model_output_dir = base_output_dir / model_type
        model_output_dir.mkdir(exist_ok=True)
        
        # Store results by genotype
        genotype_results = {
            'P9RT': {},
            'P9LT': {},
            'BPN': {}
        }
        
        # Process each leg
        for leg_name in legs:
            try:
                genotype_data = process_leg_predictions(leg_name, model_type)
                if genotype_data is not None:
                    for genotype in ['P9RT', 'P9LT', 'BPN']:
                        if genotype_data[genotype]['trials']:
                            # Store leg data for this genotype
                            genotype_results[genotype][leg_name] = {
                                'actual_stance': genotype_data[genotype]['actual_stance'],
                                'actual_swing': genotype_data[genotype]['actual_swing'],
                                'pred_stance': genotype_data[genotype]['pred_stance'],
                                'pred_swing': genotype_data[genotype]['pred_swing']
                            }
            except Exception as e:
                print(f"Error processing {leg_name} for {model_type}: {str(e)}")
                continue
        
        # Create genotype-specific directories and plots
        for genotype, leg_data in genotype_results.items():
            if leg_data:  # Only if we have data for this genotype
                # Create genotype directory
                genotype_dir = model_output_dir / genotype
                genotype_dir.mkdir(exist_ok=True)
                
                # Create summary plot
                plt.figure(figsize=(20, 8))
                plt.suptitle(f'{genotype} Phase Durations - {model_type}', fontsize=16)
                
                # Prepare data for stance phase
                stance_data = []
                stance_labels = []
                for leg_name in sorted(leg_data.keys()):  # Sort legs for consistent order
                    phase_data = leg_data[leg_name]
                    stance_data.extend([
                        phase_data['actual_stance'],
                        phase_data['pred_stance']
                    ])
                    stance_labels.extend([
                        f'{leg_name}\nActual',
                        f'{leg_name}\nPred'
                    ])
                
                # Stance phase comparison
                plt.subplot(1, 2, 1)
                bp = plt.boxplot(stance_data, labels=stance_labels, patch_artist=True)
                
                # Color boxes
                for i in range(0, len(bp['boxes']), 2):
                    bp['boxes'][i].set_facecolor('lightblue')  # Actual
                    bp['boxes'][i+1].set_facecolor('lightgreen')  # Predicted
                
                plt.title('Stance Phase Duration')
                plt.ylabel('Duration (frames)')
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45, ha='right')
                
                # Add legend
                plt.plot([], [], 'lightblue', label='Actual')
                plt.plot([], [], 'lightgreen', label='Predicted')
                plt.legend()
                
                # Prepare data for swing phase
                swing_data = []
                swing_labels = []
                for leg_name in sorted(leg_data.keys()):  # Sort legs for consistent order
                    phase_data = leg_data[leg_name]
                    swing_data.extend([
                        phase_data['actual_swing'],
                        phase_data['pred_swing']
                    ])
                    swing_labels.extend([
                        f'{leg_name}\nActual',
                        f'{leg_name}\nPred'
                    ])
                
                # Swing phase comparison
                plt.subplot(1, 2, 2)
                bp = plt.boxplot(swing_data, labels=swing_labels, patch_artist=True)
                
                # Color boxes
                for i in range(0, len(bp['boxes']), 2):
                    bp['boxes'][i].set_facecolor('lightblue')  # Actual
                    bp['boxes'][i+1].set_facecolor('lightgreen')  # Predicted
                
                plt.title('Swing Phase Duration')
                plt.ylabel('Duration (frames)')
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45, ha='right')
                
                # Add legend
                plt.plot([], [], 'lightblue', label='Actual')
                plt.plot([], [], 'lightgreen', label='Predicted')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(genotype_dir / 'phase_durations.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Save statistics
                stats = {}
                for leg_name, phase_data in leg_data.items():
                    stats[leg_name] = {
                        'actual_stance': {
                            'mean': float(np.mean(phase_data['actual_stance'])),
                            'std': float(np.std(phase_data['actual_stance'])),
                            'median': float(np.median(phase_data['actual_stance'])),
                            'min': float(np.min(phase_data['actual_stance'])),
                            'max': float(np.max(phase_data['actual_stance']))
                        },
                        'actual_swing': {
                            'mean': float(np.mean(phase_data['actual_swing'])),
                            'std': float(np.std(phase_data['actual_swing'])),
                            'median': float(np.median(phase_data['actual_swing'])),
                            'min': float(np.min(phase_data['actual_swing'])),
                            'max': float(np.max(phase_data['actual_swing']))
                        },
                        'predicted_stance': {
                            'mean': float(np.mean(phase_data['pred_stance'])),
                            'std': float(np.std(phase_data['pred_stance'])),
                            'median': float(np.median(phase_data['pred_stance'])),
                            'min': float(np.min(phase_data['pred_stance'])),
                            'max': float(np.max(phase_data['pred_stance']))
                        },
                        'predicted_swing': {
                            'mean': float(np.mean(phase_data['pred_swing'])),
                            'std': float(np.std(phase_data['pred_swing'])),
                            'median': float(np.median(phase_data['pred_swing'])),
                            'min': float(np.min(phase_data['pred_swing'])),
                            'max': float(np.max(phase_data['pred_swing']))
                        }
                    }
                
                with open(genotype_dir / 'phase_stats.json', 'w') as f:
                    json.dump(stats, f, indent=4)
        
        # Create model summary plot comparing genotypes
        plt.figure(figsize=(20, 8))
        plt.suptitle(f'Phase Durations by Genotype - {model_type}', fontsize=16)
        
        # Prepare data for stance phase comparison
        stance_data = []
        stance_labels = []
        colors = {
            'P9RT': 'lightblue',
            'P9LT': 'lightgreen',
            'BPN': 'lightpink'
        }
        box_colors = []
        
        for genotype in ['P9RT', 'P9LT', 'BPN']:
            if genotype in genotype_results and genotype_results[genotype]:
                for leg_name in sorted(genotype_results[genotype].keys()):
                    phase_data = genotype_results[genotype][leg_name]
                    stance_data.extend([
                        phase_data['actual_stance'],
                        phase_data['pred_stance']
                    ])
                    stance_labels.extend([
                        f'{genotype}\n{leg_name}\nActual',
                        f'{genotype}\n{leg_name}\nPred'
                    ])
                    box_colors.extend([colors[genotype], colors[genotype]])
        
        # Stance phase comparison
        plt.subplot(1, 2, 1)
        if stance_data:  # Only create plot if we have data
            bp = plt.boxplot(stance_data, labels=stance_labels, patch_artist=True)
            
            # Color boxes
            for box, color in zip(bp['boxes'], box_colors):
                box.set_facecolor(color)
            
            plt.title('Stance Phase Duration')
            plt.ylabel('Duration (frames)')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            
            # Add legend
            for genotype, color in colors.items():
                plt.plot([], [], color=color, label=genotype)
            plt.legend()
        
        # Prepare data for swing phase comparison
        swing_data = []
        swing_labels = []
        box_colors = []
        
        for genotype in ['P9RT', 'P9LT', 'BPN']:
            if genotype in genotype_results and genotype_results[genotype]:
                for leg_name in sorted(genotype_results[genotype].keys()):
                    phase_data = genotype_results[genotype][leg_name]
                    swing_data.extend([
                        phase_data['actual_swing'],
                        phase_data['pred_swing']
                    ])
                    swing_labels.extend([
                        f'{genotype}\n{leg_name}\nActual',
                        f'{genotype}\n{leg_name}\nPred'
                    ])
                    box_colors.extend([colors[genotype], colors[genotype]])
        
        # Swing phase comparison
        plt.subplot(1, 2, 2)
        if swing_data:  # Only create plot if we have data
            bp = plt.boxplot(swing_data, labels=swing_labels, patch_artist=True)
            
            # Color boxes
            for box, color in zip(bp['boxes'], box_colors):
                box.set_facecolor(color)
            
            plt.title('Swing Phase Duration')
            plt.ylabel('Duration (frames)')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            
            # Add legend
            for genotype, color in colors.items():
                plt.plot([], [], color=color, label=genotype)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(model_output_dir / 'all_genotypes_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save model summary statistics
        model_stats = {}
        for genotype, leg_data in genotype_results.items():
            if leg_data:  # Only if we have data for this genotype
                model_stats[genotype] = {}
                for leg_name, phase_data in leg_data.items():
                    model_stats[genotype][leg_name] = {
                        'actual_stance': {
                            'mean': float(np.mean(phase_data['actual_stance'])),
                            'std': float(np.std(phase_data['actual_stance']))
                        },
                        'actual_swing': {
                            'mean': float(np.mean(phase_data['actual_swing'])),
                            'std': float(np.std(phase_data['actual_swing']))
                        },
                        'predicted_stance': {
                            'mean': float(np.mean(phase_data['pred_stance'])),
                            'std': float(np.std(phase_data['pred_stance']))
                        },
                        'predicted_swing': {
                            'mean': float(np.mean(phase_data['pred_swing'])),
                            'std': float(np.std(phase_data['pred_swing']))
                        }
                    }
        
        with open(model_output_dir / 'model_summary_stats.json', 'w') as f:
            json.dump(model_stats, f, indent=4)
    
    print("\nProcessing completed!")

if __name__ == "__main__":
    main() 