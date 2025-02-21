import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy import signal
from pathlib import Path

def filter_frames(df):
    """Filter frames to only include frames 400-1000 from each trial."""
    # Get trial indices (assuming 1400 frames per trial)
    trial_size = 1400
    num_trials = len(df) // trial_size
    
    # Create filtered dataframe
    filtered_rows = []
    for trial in range(num_trials):
        start_idx = trial * trial_size + 400
        end_idx = trial * trial_size + 1000
        filtered_rows.append(df.iloc[start_idx:end_idx])
    
    # Concatenate all filtered trials
    filtered_df = pd.concat(filtered_rows, axis=0, ignore_index=True)
    print(f"Filtered data from {len(df)} to {len(filtered_df)} frames")
    return filtered_df

def load_data(file_path, genotype=None):
    """Load and preprocess the dataset from a CSV file."""
    try:
        # Load the full data
        df = pd.read_csv(file_path)
        print(f"Raw data loaded successfully with shape: {df.shape}")
        
        # Filter for genotype
        if genotype:
            df = df[df['genotype'] == genotype]
            print(f"Filtered for {genotype} genotype, new shape: {df.shape}")
        
        # Verify velocity columns exist
        required_cols = ['x_vel', 'y_vel', 'z_vel']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required velocity columns: {missing_cols}")
        
        # Create trial IDs based on frame numbers
        # Each trial should be 1400 frames
        trial_size = 1400
        num_trials = len(df) // trial_size
        df['trial_id'] = np.repeat(range(num_trials), trial_size)[:len(df)]
        
        # Print trial statistics before filtering
        trial_lengths = df.groupby('trial_id').size()
        print("\nTrial statistics before filtering:")
        print(f"Number of trials: {len(trial_lengths)}")
        print(f"Trial length statistics:")
        print(trial_lengths.describe())
        
        # Filter frames 400-1000 from each trial
        filtered_rows = []
        skipped_trials = []
        short_trials = []
        
        for trial in range(num_trials):
            start_idx = trial * trial_size
            end_idx = (trial + 1) * trial_size
            
            if end_idx > len(df):  # Skip incomplete trials at the end
                short_trials.append((trial, len(df) - start_idx))
                continue
            
            try:
                trial_data = df.iloc[start_idx:end_idx]
                filtered_rows.append(trial_data.iloc[400:1000])
            except Exception as e:
                skipped_trials.append((trial, str(e)))
        
        # Print filtering statistics
        print("\nFiltering statistics:")
        print(f"Total trials: {num_trials}")
        print(f"Trials included: {len(filtered_rows)}")
        if short_trials:
            print("\nSkipped trials (too short):")
            for trial, length in short_trials:
                print(f"Trial {trial}: {length} frames")
        if skipped_trials:
            print("\nSkipped trials (errors):")
            for trial, error in skipped_trials:
                print(f"Trial {trial}: {error}")
        
        # Concatenate all filtered trials
        df = pd.concat(filtered_rows, axis=0, ignore_index=True)
        print(f"\nFiltered to frames 400-1000, new shape: {df.shape}")
        
        return df
        
    except Exception as e:
        print(f"Error loading/preprocessing data: {e}")
        raise

def calculate_enhanced_features(df):
    """Calculate enhanced features including velocities and their derivatives."""
    # Initialize features dictionary
    features_dict = {}
    
    # Get trial size and number of trials
    trial_size = 600  # After filtering (1000 - 400 = 600)
    num_trials = len(df) // trial_size
    
    # Process each trial separately
    for trial in range(num_trials):
        start_idx = trial * trial_size
        end_idx = start_idx + trial_size
        
        # Get trial data
        trial_data = df.iloc[start_idx:end_idx]
        
        # Original velocities for this trial
        if trial == 0:  # Initialize arrays on first trial
            features_dict['x_vel'] = np.zeros(len(df))
            features_dict['y_vel'] = np.zeros(len(df))
            features_dict['z_vel'] = np.zeros(len(df))
        
        features_dict['x_vel'][start_idx:end_idx] = trial_data['x_vel'].values
        features_dict['y_vel'][start_idx:end_idx] = trial_data['y_vel'].values
        features_dict['z_vel'][start_idx:end_idx] = trial_data['z_vel'].values
        
        # Calculate derivatives (acceleration) for this trial
        dt = 1/200  # 200Hz sampling rate
        for coord in ['x', 'y', 'z']:
            vel = trial_data[f'{coord}_vel'].values
            acc = np.zeros_like(vel)
            acc[1:-1] = (vel[2:] - vel[:-2]) / (2 * dt)
            
            # Initialize array on first trial
            if trial == 0:
                features_dict[f'{coord}_acc'] = np.zeros(len(df))
            
            features_dict[f'{coord}_acc'][start_idx:end_idx] = acc
        
        # Calculate velocity magnitude for this trial
        vel_mag = np.sqrt(
            trial_data['x_vel'].values**2 + 
            trial_data['y_vel'].values**2 + 
            trial_data['z_vel'].values**2
        )
        
        # Initialize array on first trial
        if trial == 0:
            features_dict['velocity_magnitude'] = np.zeros(len(df))
        
        features_dict['velocity_magnitude'][start_idx:end_idx] = vel_mag
        
        # Calculate acceleration magnitude for this trial
        acc_mag = np.sqrt(
            features_dict['x_acc'][start_idx:end_idx]**2 + 
            features_dict['y_acc'][start_idx:end_idx]**2 + 
            features_dict['z_acc'][start_idx:end_idx]**2
        )
        
        # Initialize array on first trial
        if trial == 0:
            features_dict['acceleration_magnitude'] = np.zeros(len(df))
        
        features_dict['acceleration_magnitude'][start_idx:end_idx] = acc_mag
    
    # Convert to DataFrame all at once
    features = pd.DataFrame(features_dict, index=df.index)
    
    print(f"Enhanced features shape: {features.shape}")
    return features

def calculate_psd_features(data, fs=200):
    """
    Calculate power spectral density features for velocity data up to Nyquist frequency.
    
    Args:
        data (numpy.ndarray): Time series data
        fs (int): Sampling frequency in Hz
        
    Returns:
        dict: Dictionary containing PSD features
    """
    # Calculate PSD using Welch's method up to Nyquist frequency (fs/2 = 100 Hz)
    frequencies, psd = signal.welch(data, fs=fs, nperseg=fs)
    
    # Calculate features from PSD
    total_power = np.sum(psd)
    peak_frequency = frequencies[np.argmax(psd)]
    mean_frequency = np.sum(frequencies * psd) / total_power
    
    # Calculate power in different frequency bands
    # More granular frequency bands up to 200 Hz
    bands = {
        'very_low': (0, 10),    # 0-10 Hz: Very slow movements
        'low': (10, 30),        # 10-30 Hz: Slow movements
        'medium': (30, 60),     # 30-60 Hz: Medium speed movements
        'high': (60, 100),      # 60-100 Hz: Fast movements
        'very_high': (100, 200) # 100-200 Hz: Very fast movements/noise
    }
    
    def get_band_power(band):
        mask = (frequencies >= band[0]) & (frequencies < band[1])
        return np.sum(psd[mask])
    
    # Calculate power in each band
    band_powers = {
        f'{band_name}_power': get_band_power(freq_range)
        for band_name, freq_range in bands.items()
    }
    
    # Calculate relative power (percentage of total power in each band)
    relative_powers = {
        f'{band_name}_relative_power': power / total_power
        for band_name, power in band_powers.items()
    }
    
    # Combine all features
    features = {
        'total_power': total_power,
        'peak_freq': peak_frequency,
        'mean_freq': mean_frequency,
        **band_powers,
        **relative_powers
    }
    
    return features

def calculate_lagged_features(df, cols, lags, trial_size):
    """Calculate lagged features (both future and past) for given columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        cols (list): List of columns to calculate lags for
        lags (list): List of lag values (positive for future, negative for past)
        trial_size (int): Size of each trial to avoid crossing trial boundaries
        
    Returns:
        pd.DataFrame: DataFrame with lagged features
    """
    lagged_features = {}
    num_trials = len(df) // trial_size
    
    for col in cols:
        for lag in lags:
            # Create feature name
            direction = 'future' if lag > 0 else 'past'
            abs_lag = abs(lag)
            feature_name = f"{col}_{direction}_{abs_lag}"
            
            # Initialize array
            lagged_values = np.zeros(len(df))
            
            # Calculate lags for each trial separately
            for trial in range(num_trials):
                start_idx = trial * trial_size
                end_idx = start_idx + trial_size
                trial_data = df[col].iloc[start_idx:end_idx].values
                
                if lag > 0:  # Future values
                    lagged_values[start_idx:end_idx-lag] = trial_data[lag:]
                    lagged_values[end_idx-lag:end_idx] = trial_data[-1]  # Pad with last value
                else:  # Past values
                    lagged_values[start_idx:start_idx-lag] = trial_data[0]  # Pad with first value
                    lagged_values[start_idx-lag:end_idx] = trial_data[:lag]
            
            lagged_features[feature_name] = lagged_values
    
    return pd.DataFrame(lagged_features, index=df.index)

def calculate_spearman_correlation(data_path, genotype, output_dir='spearman_correlation_analysis'):
    """
    Calculate Spearman's rank correlation coefficient between variables in the dataset.
    
    Args:
        data_path (str): Path to the CSV file containing the data
        genotype (str): Genotype to analyze ('P9RT', 'BPN', or 'P9LT')
        output_dir (str): Directory to save the correlation results and plots
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir) / genotype
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    print(f"\nLoading and preprocessing data for {genotype} from: {data_path}")
    df = load_data(data_path, genotype)
    
    # Calculate enhanced features
    print("\nCalculating enhanced features (velocities and their derivatives)...")
    enhanced_features = calculate_enhanced_features(df)
    
    # Add enhanced features to the original dataframe FIRST
    for col in enhanced_features.columns:
        if col not in df.columns:
            df[col] = enhanced_features[col]
    
    # Calculate moving averages for velocities
    print("\nCalculating moving averages for velocities...")
    velocity_cols = ['x_vel', 'y_vel', 'z_vel']
    windows = [5, 10, 20]  # Window sizes for moving averages
    
    # Each trial has 600 frames after filtering (1000-400)
    trial_size = 600
    num_trials = len(df) // trial_size
    print(f"\nNumber of trials after filtering: {num_trials}")
    
    # Calculate moving averages
    for window in windows:
        for col in velocity_cols:
            ma_values = []
            for trial in range(num_trials):
                start_idx = trial * trial_size
                end_idx = start_idx + trial_size
                trial_data = df[col].iloc[start_idx:end_idx]
                # Calculate moving average within the trial
                ma = trial_data.rolling(window=window, center=True, min_periods=1).mean()
                ma_values.extend(ma.tolist())
            df[f"{col}_ma{window}"] = ma_values
    
    # Now collect all velocity-related features that exist in the DataFrame
    velocity_related_features = (
        velocity_cols +  # Basic velocities
        [f"{col}_ma{window}" for col in velocity_cols for window in windows] +  # Moving averages
        [f"{coord}_acc" for coord in ['x', 'y', 'z']] +  # Accelerations
        ['velocity_magnitude', 'acceleration_magnitude']  # Magnitudes
    )
    
    # Verify all features exist before calculating lags
    missing_features = [feat for feat in velocity_related_features if feat not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features before lag calculation: {missing_features}")
    
    # Calculate lagged features for all velocity-related features
    print("\nCalculating lagged features for all velocity-related features...")
    lag_values = [5, 10, 20]  # Future lags
    
    # Calculate future values for all velocity-related features
    lagged_features = calculate_lagged_features(df, velocity_related_features, lag_values, trial_size)
    
    # Add lagged features to dataframe
    for col in lagged_features.columns:
        df[col] = lagged_features[col]
    
    # Calculate PSD features for each velocity component
    print("\nCalculating power spectral density features...")
    psd_features = {}
    for col in velocity_cols:
        for trial in range(num_trials):
            start_idx = trial * trial_size
            end_idx = start_idx + trial_size
            trial_data = df[col].iloc[start_idx:end_idx].values
            
            # Calculate PSD features for this trial
            features = calculate_psd_features(trial_data)
            
            # Store features for this trial
            for feature_name, value in features.items():
                feature_key = f"{col}_{feature_name}"
                if feature_key not in psd_features:
                    psd_features[feature_key] = np.zeros(len(df))
                psd_features[feature_key][start_idx:end_idx] = value
    
    # Add PSD features to dataframe
    psd_df = pd.DataFrame(psd_features, index=df.index)
    df = pd.concat([df, psd_df], axis=1)
    
    # Print shape information for debugging
    print(f"\nShape of data after adding all features: {df.shape}")
    print("\nPSD features added:")
    for col in psd_df.columns:
        print(f"  {col}")
    
    # Separate features and joint angles
    joint_cols = [col for col in df.columns if col.endswith(('_flex', '_rot', '_abduct'))]
    
    # Define base velocity features
    base_velocity_features = [
        'x_vel', 'y_vel', 'z_vel',  # Raw velocities
        'x_vel_ma5', 'y_vel_ma5', 'z_vel_ma5',  # Moving averages
        'x_vel_ma10', 'y_vel_ma10', 'z_vel_ma10',
        'x_vel_ma20', 'y_vel_ma20', 'z_vel_ma20',
        'x_acc', 'y_acc', 'z_acc',  # Accelerations
        'velocity_magnitude', 'acceleration_magnitude'  # Magnitudes
    ]
    
    # Add lagged features for all velocity-related features
    lagged_features = [
        f"{feat}_future_{lag}" 
        for feat in velocity_related_features 
        for lag in lag_values
    ]
    
    # Add PSD features (use actual column names from DataFrame)
    psd_features = [col for col in psd_df.columns]
    
    # Combine all features
    feature_cols = base_velocity_features + lagged_features + psd_features
    
    print(f"\nAnalyzing correlations between {len(feature_cols)} velocity features and {len(joint_cols)} joint angles")
    print("\nVelocity features being analyzed:")
    for feat in feature_cols:
        print(f"  {feat}")
    
    # Calculate Spearman correlation
    print("\nCalculating Spearman correlation coefficients...")
    
    # Initialize correlation matrix
    correlation_matrix = pd.DataFrame(
        np.zeros((len(feature_cols), len(joint_cols))),
        index=feature_cols,
        columns=joint_cols
    )
    
    # Calculate correlations across all trials at once
    for feature in feature_cols:
        for joint in joint_cols:
            try:
                # Get all data for this feature-joint pair
                x = df[feature].values
                y = df[joint].values
                
                # Remove any NaN values
                mask = ~(np.isnan(x) | np.isnan(y))
                x = x[mask]
                y = y[mask]
                
                # Calculate correlation if we have enough data
                if len(x) > 0 and len(y) > 0:
                    correlation, p_value = stats.spearmanr(x, y)
                    correlation_matrix.loc[feature, joint] = correlation
                    
            except Exception as e:
                print(f"\nError calculating correlation between {feature} and {joint}: {e}")
                continue
    
    # Save full correlation matrix to CSV
    correlation_csv = output_path / f'spearman_correlation_matrix_{genotype}.csv'
    correlation_matrix.to_csv(correlation_csv)
    print(f"Saved correlation matrix to: {correlation_csv}")
    
    # Create correlation heatmap for features vs joint angles
    plt.figure(figsize=(24, 12))
    sns.heatmap(correlation_matrix, 
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.2f',
                vmin=-1,
                vmax=1,
                cbar_kws={"shrink": .8})
    
    plt.title(f"Spearman's Rank Correlation: All Features vs Joint Angles ({genotype})")
    plt.xlabel('Joint Angles')
    plt.ylabel('Velocity Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save heatmap
    heatmap_path = output_path / f'spearman_correlation_heatmap_{genotype}.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation heatmap to: {heatmap_path}")
    
    # Find significant correlations
    print("\nAnalyzing significant correlations...")
    significant_correlations = []
    
    # Check all correlations for significance
    for feature in feature_cols:
        for joint in joint_cols:
            try:
                x = df[feature].values
                y = df[joint].values
                
                # Remove any NaN values
                mask = ~(np.isnan(x) | np.isnan(y))
                x = x[mask]
                y = y[mask]
                
                # Calculate correlation and p-value if we have enough data
                if len(x) > 0 and len(y) > 0:
                    correlation, p_value = stats.spearmanr(x, y)
                    if abs(correlation) > 0.3 and p_value < 0.05:  # Adjusted threshold to 0.3
                        significant_correlations.append({
                            'Feature': feature,
                            'Joint Angle': joint,
                            'Correlation': correlation,
                            'P-value': p_value
                        })
            except Exception as e:
                print(f"\nError analyzing correlation between {feature} and {joint}: {e}")
                continue
    
    # Sort by absolute correlation strength
    significant_correlations.sort(key=lambda x: abs(x['Correlation']), reverse=True)
    
    # Save significant correlations to CSV
    if significant_correlations:
        sig_corr_df = pd.DataFrame(significant_correlations)
        sig_corr_path = output_path / f'significant_correlations_{genotype}.csv'
        sig_corr_df.to_csv(sig_corr_path, index=False)
        print(f"Saved significant correlations to: {sig_corr_path}")
        
        # Print top correlations
        print(f"\nTop 20 strongest correlations for {genotype}:")
        print(sig_corr_df.head(20).to_string(index=False))
    else:
        print(f"\nNo significant correlations found for {genotype}.")
    
    return correlation_matrix

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
        
        parent_dir = Path("spearman_correlation_results")
        parent_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each genotype
        genotypes = ['P9RT', 'BPN', 'P9LT']
        
        for genotype in genotypes:
            print(f"\nProcessing {genotype} genotype...")
            output_dir = parent_dir / genotype
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Calculate correlations and generate plots
            calculate_spearman_correlation(file_path, genotype, output_dir)
            print(f"\nResults saved to {output_dir}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data file is available in one of the expected locations.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 