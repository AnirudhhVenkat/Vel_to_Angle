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
    
    # Use the same random trial assignment as in training
    np.random.seed(42)  # Same seed as in training
    trial_indices = np.random.permutation(num_trials)
    
    # Sort predictions and targets by trial indices
    predictions = predictions[trial_indices]
    targets = targets[trial_indices]
    
    print(f"Using random trial assignment with {num_trials} trials")
    print(f"Random trial order: {trial_indices}")  # Show actual permutation
    
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

def process_leg_predictions(leg_name, results_dir='lstm_results'):
    """Process predictions for B_flex angle of a specific leg."""
    print(f"\nProcessing {leg_name}B_flex predictions...")
    
    # Find the leg directory
    leg_dir = Path(results_dir) / leg_name
    if not leg_dir.exists():
        print(f"No results directory found for {leg_name}")
        return None, None, None, None
    
    # Look for predictions file
    predictions_path = leg_dir / 'plots' / 'predictions' / 'trial_predictions.npz'
    if not predictions_path.exists():
        print(f"No predictions found for {leg_name}")
        return None, None, None, None
    
    print(f"Found predictions at: {predictions_path}")
    
    # Create output directory
    output_dir = leg_dir / 'stance_swing_analysis'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # Load predictions
        predictions, targets, output_features, frames = load_predictions(predictions_path)
        num_trials = predictions.shape[0]
        
        # Find B_flex index
        b_flex_idx = None
        for i, feature in enumerate(output_features):
            if feature.endswith('B_flex'):
                b_flex_idx = i
                break
        
        if b_flex_idx is None:
            print(f"No B_flex angle found for {leg_name}")
            return None, None, None, None
        
        print(f"Found {num_trials} trials")
        
        # Initialize lists to store phase durations
        actual_stance_durations = []
        actual_swing_durations = []
        pred_stance_durations = []
        pred_swing_durations = []
        
        # Create PDF for all trials
        pdf_path = output_dir / f'{leg_name}_B_flex_all_trials.pdf'
        with PdfPages(pdf_path) as pdf:
            # First page: Summary statistics
            plt.figure(figsize=(12, 8))
            plt.axis('off')
            plt.text(0.1, 0.95, f'{leg_name}B_flex Phase Analysis', fontsize=16, fontweight='bold')
            plt.text(0.1, 0.85, f'Number of trials: {num_trials}', fontsize=12)
            plt.text(0.1, 0.75, 'Color coding:', fontsize=12)
            plt.text(0.1, 0.7, '  Green: Stance phase (leading to peak)', fontsize=12)
            plt.text(0.1, 0.65, '  Blue: Swing phase (after peak)', fontsize=12)
            plt.text(0.1, 0.6, '  Black triangles: Detected peaks', fontsize=12)
            plt.text(0.1, 0.55, '  Blue line: Actual angle', fontsize=12)
            plt.text(0.1, 0.5, '  Red dashed line: Predicted angle', fontsize=12)
            pdf.savefig()
            plt.close()
            
            # Process each trial
            for trial_idx in range(num_trials):
                print(f"Processing trial {trial_idx + 1}/{num_trials}")
                
                # Get B_flex predictions and targets
                pred = predictions[trial_idx, :, b_flex_idx]
                target = targets[trial_idx, :, b_flex_idx]
                
                # Skip if either predictions or targets contain NaN values
                if np.isnan(pred).any() or np.isnan(target).any():
                    print(f"Skipping trial {trial_idx + 1} due to NaN values")
                    continue
                
                # Create trial plot
                plot_trial_phases(pred, target, frames, trial_idx, leg_name, pdf)
                
                # Calculate phase durations for actual data
                actual_phases, _ = detect_phases(target)
                trial_actual_stance, trial_actual_swing = calculate_phase_durations(actual_phases)
                actual_stance_durations.extend(trial_actual_stance)
                actual_swing_durations.extend(trial_actual_swing)
                
                # Calculate phase durations for predicted data
                pred_phases, _ = detect_phases(pred)
                trial_pred_stance, trial_pred_swing = calculate_phase_durations(pred_phases)
                pred_stance_durations.extend(trial_pred_stance)
                pred_swing_durations.extend(trial_pred_swing)
            
            # Add final page with phase duration statistics
            plt.figure(figsize=(12, 8))
            plt.axis('off')
            plt.text(0.1, 0.95, f'{leg_name}B_flex Phase Duration Statistics', fontsize=16, fontweight='bold')
            
            # Calculate statistics for both actual and predicted
            stats = {
                'actual_stance': {
                    'mean': float(np.mean(actual_stance_durations)),
                    'std': float(np.std(actual_stance_durations)),
                    'median': float(np.median(actual_stance_durations)),
                    'min': float(np.min(actual_stance_durations)),
                    'max': float(np.max(actual_stance_durations))
                },
                'actual_swing': {
                    'mean': float(np.mean(actual_swing_durations)),
                    'std': float(np.std(actual_swing_durations)),
                    'median': float(np.median(actual_swing_durations)),
                    'min': float(np.min(actual_swing_durations)),
                    'max': float(np.max(actual_swing_durations))
                },
                'predicted_stance': {
                    'mean': float(np.mean(pred_stance_durations)),
                    'std': float(np.std(pred_stance_durations)),
                    'median': float(np.median(pred_stance_durations)),
                    'min': float(np.min(pred_stance_durations)),
                    'max': float(np.max(pred_stance_durations))
                },
                'predicted_swing': {
                    'mean': float(np.mean(pred_swing_durations)),
                    'std': float(np.std(pred_swing_durations)),
                    'median': float(np.median(pred_swing_durations)),
                    'min': float(np.min(pred_swing_durations)),
                    'max': float(np.max(pred_swing_durations))
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
        with open(output_dir / 'phase_duration_stats.json', 'w') as f:
            json.dump({f'{leg_name}B_flex': stats}, f, indent=4)
        
        print(f"PDF and statistics saved to {output_dir}")
        return actual_stance_durations, actual_swing_durations, pred_stance_durations, pred_swing_durations
    
    except Exception as e:
        print(f"Error processing {leg_name} for {results_dir}: {str(e)}")
        return None, None, None, None

def main():
    """Process B_flex predictions for all legs."""
    # Define legs to process
    legs = ['R1', 'L1', 'R2', 'L2', 'R3', 'L3']
    
    # Define model types to check
    model_types = ['lstm_results', 'transformer_results', 'tcn_results', 'hybrid_results']
    
    # Store all phase durations
    all_actual_stance = []
    all_actual_swing = []
    all_pred_stance = []
    all_pred_swing = []
    processed_legs = []
    
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
        model_output_dir = base_output_dir / model_type
        model_output_dir.mkdir(exist_ok=True)
        
        model_actual_stance = []
        model_actual_swing = []
        model_pred_stance = []
        model_pred_swing = []
        model_processed_legs = []
        
        for leg_name in legs:
            try:
                actual_stance, actual_swing, pred_stance, pred_swing = process_leg_predictions(leg_name, model_type)
                if actual_stance is not None:
                    model_actual_stance.append(actual_stance)
                    model_actual_swing.append(actual_swing)
                    model_pred_stance.append(pred_stance)
                    model_pred_swing.append(pred_swing)
                    model_processed_legs.append(leg_name)
            except Exception as e:
                print(f"Error processing {leg_name} for {model_type}: {str(e)}")
                continue
        
        # Create combined box plot for this model type if we have data
        if len(model_processed_legs) > 1:
            plot_phase_durations(model_actual_stance, model_actual_swing,
                               model_pred_stance, model_pred_swing,
                               model_processed_legs,
                               model_output_dir / 'combined_phase_durations_boxplot.png')
            
            # Save combined statistics
            combined_stats = {
                'actual_stance': {leg: np.mean(durations) for leg, durations in zip(model_processed_legs, model_actual_stance)},
                'actual_swing': {leg: np.mean(durations) for leg, durations in zip(model_processed_legs, model_actual_swing)},
                'predicted_stance': {leg: np.mean(durations) for leg, durations in zip(model_processed_legs, model_pred_stance)},
                'predicted_swing': {leg: np.mean(durations) for leg, durations in zip(model_processed_legs, model_pred_swing)}
            }
            
            with open(model_output_dir / 'combined_phase_stats.json', 'w') as f:
                json.dump(combined_stats, f, indent=4)
            
            # Add to overall results
            all_actual_stance.extend(model_actual_stance)
            all_actual_swing.extend(model_actual_swing)
            all_pred_stance.extend(model_pred_stance)
            all_pred_swing.extend(model_pred_swing)
            processed_legs.extend([f"{leg}_{model_type}" for leg in model_processed_legs])
    
    # Create overall combined box plot if we have data from multiple models
    if len(processed_legs) > 1:
        plot_phase_durations(all_actual_stance, all_actual_swing,
                           all_pred_stance, all_pred_swing,
                           processed_legs,
                           base_output_dir / 'overall_combined_phase_durations_boxplot.png')
        
        # Save overall combined statistics
        overall_combined_stats = {
            'actual_stance': {leg: np.mean(durations) for leg, durations in zip(processed_legs, all_actual_stance)},
            'actual_swing': {leg: np.mean(durations) for leg, durations in zip(processed_legs, all_actual_swing)},
            'predicted_stance': {leg: np.mean(durations) for leg, durations in zip(processed_legs, all_pred_stance)},
            'predicted_swing': {leg: np.mean(durations) for leg, durations in zip(processed_legs, all_pred_swing)}
        }
        
        with open(base_output_dir / 'overall_combined_phase_stats.json', 'w') as f:
            json.dump(overall_combined_stats, f, indent=4)
    
    print("\nProcessing completed!")

if __name__ == "__main__":
    main() 