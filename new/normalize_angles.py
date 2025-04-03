#!/usr/bin/env python3
"""
Script to normalize abduct and rot angles in fly datasets by ensuring consistent sign.

This script:
1. Loads the fly datasets
2. For each trial, determines the dominant sign (positive or negative) for abduct and rot angles
3. Normalizes all values within each trial to be consistent with the dominant sign
4. Saves the normalized data back to disk

Usage:
    python normalize_angles.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import traceback
import os
import seaborn as sns

def get_available_data_path(genotype):
    """Try multiple possible data paths and return the first available one based on genotype."""
    if genotype == 'ES':
        possible_paths = [
            r"Z:\Divya\TEMP_transfers\toAni\4_StopProjectData_forES\df_preproc_fly_centric.parquet"
        ]
    else:
        possible_paths = [
            "Z:/Divya/TEMP_transfers/toAni/BPN_P9LT_P9RT_flyCoords.csv",  # Network drive path
            "/Users/anivenkat/Downloads/BPN_P9LT_P9RT_flyCoords.csv",      # Local Mac path
            "C:/Users/bidayelab/Downloads/BPN_P9LT_P9RT_flyCoords.csv"     # Local Windows path
        ]
    
    for path in possible_paths:
        if Path(path).exists():
            print(f"Using data file: {path}")
            return path
    
    raise FileNotFoundError(f"Could not find the data file for genotype {genotype}")

def normalize_angles(data, trial_size, angle_cols, frame_range=None):
    """
    Normalize the sign of angle data for each trial.
    
    Args:
        data: DataFrame containing angle data
        trial_size: Number of frames per trial
        angle_cols: List of angle columns to normalize (abduct and rot angles)
        frame_range: Tuple of (start_frame, end_frame) to consider for visualization highlighting
                     (no longer used for determining dominant sign)
        
    Returns:
        DataFrame with normalized angle data
    """
    num_trials = len(data) // trial_size
    print(f"Processing {num_trials} trials with {trial_size} frames each")
    print(f"NOTE: Using all frames to determine dominant sign based on majority count")
    print(f"      Values will be flipped to match the majority sign (negative or positive)")
    
    # Create copy of data to avoid modifying the original
    normalized_data = data.copy()
    
    # Setup plot storage
    plot_dir = Path("angle_normalization_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Statistics storage
    stats = {
        'genotype': [],
        'trial': [],
        'angle': [],
        'dominant_sign': [],
        'positive_count': [],
        'negative_count': [],
        'zero_count': [],
        'mean_before': [],
        'mean_after': [],
        'stddev_before': [],
        'stddev_after': []
    }
    
    # Process each trial
    for trial_idx in tqdm(range(num_trials), desc="Normalizing trials"):
        trial_start = trial_idx * trial_size
        trial_end = (trial_idx + 1) * trial_size
        
        # Make sure we have enough frames for a complete trial
        if trial_end <= len(data):
            # Get the trial data
            trial_data = normalized_data.iloc[trial_start:trial_end]
            
            # Get genotype
            genotype = trial_data['genotype'].iloc[0]
            
            # Determine visualization frame range based on genotype (for highlighting only)
            if frame_range is None:
                if genotype == 'ES':
                    frame_range = (0, 650)
                else:  # BPN, P9RT, P9LT
                    frame_range = (400, 1000)
            
            start_frame, end_frame = frame_range
            
            # Adjust end_frame if it exceeds trial size
            end_frame = min(end_frame, trial_size)
            
            # Process each angle column
            for angle_col in angle_cols:
                # Use ALL frames for determining dominant sign
                angle_values = trial_data[angle_col].values
                
                # Skip if no valid values
                if len(angle_values) == 0 or np.isnan(angle_values).all():
                    continue
                
                # Count positive, negative, and zero values (excluding NaN)
                valid_values = angle_values[~np.isnan(angle_values)]
                positive_count = np.sum(valid_values > 0)
                negative_count = np.sum(valid_values < 0)
                zero_count = np.sum(valid_values == 0)
                
                # Determine dominant sign based on counts
                if positive_count > negative_count:
                    dominant_sign = 1  # Positive is dominant
                elif negative_count > positive_count:
                    dominant_sign = -1  # Negative is dominant
                else:
                    # If counts are equal, use mean for tie-breaking
                    mean_value = np.nanmean(angle_values)
                    dominant_sign = np.sign(mean_value) if mean_value != 0 else 1
                
                # Statistics before normalization
                mean_before = np.nanmean(angle_values)
                stddev_before = np.nanstd(angle_values)
                
                # Store original values for the plot
                original_values = trial_data[angle_col].values.copy()
                
                # Normalize sign to match the dominant sign
                sign_flipped = False
                
                # If dominant sign is positive, make all values positive
                if dominant_sign > 0:
                    # Find values with the wrong sign (negative) and flip them
                    wrong_sign_indices = np.where(angle_values < 0)[0]
                    if len(wrong_sign_indices) > 0:
                        sign_flipped = True
                        for idx in wrong_sign_indices:
                            if not np.isnan(angle_values[idx]):  # Skip NaN values
                                normalized_data.iloc[trial_start + idx, normalized_data.columns.get_loc(angle_col)] = -angle_values[idx]
                
                # If dominant sign is negative, make all values negative
                else:  # dominant_sign < 0
                    # Find values with the wrong sign (positive) and flip them
                    wrong_sign_indices = np.where(angle_values > 0)[0]
                    if len(wrong_sign_indices) > 0:
                        sign_flipped = True
                        for idx in wrong_sign_indices:
                            if not np.isnan(angle_values[idx]):  # Skip NaN values
                                normalized_data.iloc[trial_start + idx, normalized_data.columns.get_loc(angle_col)] = -angle_values[idx]
                
                # Statistics after normalization
                normalized_angle_values = normalized_data.iloc[trial_start:trial_end][angle_col].values
                mean_after = np.nanmean(normalized_angle_values)
                stddev_after = np.nanstd(normalized_angle_values)
                
                # Store statistics
                stats['genotype'].append(genotype)
                stats['trial'].append(trial_idx)
                stats['angle'].append(angle_col)
                stats['dominant_sign'].append("negative" if dominant_sign < 0 else "positive")
                stats['positive_count'].append(positive_count)
                stats['negative_count'].append(negative_count)
                stats['zero_count'].append(zero_count)
                stats['mean_before'].append(mean_before)
                stats['mean_after'].append(mean_after)
                stats['stddev_before'].append(stddev_before)
                stats['stddev_after'].append(stddev_after)
                
                # Create visualization for this trial and angle
                if trial_idx % 10 == 0:  # Only plot every 10th trial to avoid too many plots
                    try:
                        plt.figure(figsize=(12, 6))
                        
                        # Original values
                        plt.subplot(1, 2, 1)
                        x_values = np.arange(len(original_values))
                        plt.plot(x_values, original_values)
                        plt.axhline(y=0, color='r', linestyle='-', alpha=0.5)
                        plt.axvspan(start_frame, end_frame, alpha=0.2, color='yellow', label='Highlight Region')
                        plt.title(f"Original {angle_col} - Trial {trial_idx} ({genotype})\n" + 
                                 f"Mean: {mean_before:.2f}, +: {positive_count}, -: {negative_count}")
                        plt.xlabel("Frame")
                        plt.ylabel("Angle Value")
                        plt.legend()
                        
                        # Normalized values
                        plt.subplot(1, 2, 2)
                        normalized_values = normalized_data.iloc[trial_start:trial_end, normalized_data.columns.get_loc(angle_col)].values
                        plt.plot(x_values, normalized_values)
                        plt.axhline(y=0, color='r', linestyle='-', alpha=0.5)
                        plt.axvspan(start_frame, end_frame, alpha=0.2, color='yellow', label='Highlight Region')
                        
                        if dominant_sign > 0:
                            title_text = f"Normalized {angle_col} - Trial {trial_idx} ({genotype})\n"
                            if sign_flipped:
                                title_text += f"Made all values POSITIVE (+ dominant: {positive_count} vs {negative_count})"
                            else:
                                title_text += f"Already all positive"
                        else:  # dominant_sign < 0
                            title_text = f"Normalized {angle_col} - Trial {trial_idx} ({genotype})\n"
                            if sign_flipped:
                                title_text += f"Made all values NEGATIVE (- dominant: {negative_count} vs {positive_count})"
                            else:
                                title_text += f"Already all negative"
                                
                        title_text += f"\nMean: {mean_after:.2f}"
                        plt.title(title_text)
                        plt.xlabel("Frame")
                        plt.ylabel("Angle Value")
                        plt.legend()
                        
                        plt.tight_layout()
                        file_name = f"{genotype}_trial{trial_idx}_{angle_col}"
                        if sign_flipped:
                            file_name += "_NORMALIZED"
                        plt.savefig(plot_dir / f"{file_name}.png")
                        plt.close()
                        
                        # Create a third plot showing the difference if sign was flipped
                        if sign_flipped:
                            plt.figure(figsize=(10, 6))
                            plt.plot(x_values, original_values, 'r-', label='Original')
                            plt.plot(x_values, normalized_values, 'g-', label='Normalized')
                            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                            plt.axvspan(start_frame, end_frame, alpha=0.1, color='yellow', label='Highlight Region')
                            
                            if dominant_sign > 0:
                                title = f"Sign Normalization - {angle_col} - Trial {trial_idx} ({genotype})\n" + \
                                        f"Made all values POSITIVE (+ dominant: {positive_count} vs {negative_count})"
                            else:
                                title = f"Sign Normalization - {angle_col} - Trial {trial_idx} ({genotype})\n" + \
                                        f"Made all values NEGATIVE (- dominant: {negative_count} vs {positive_count})"
                                        
                            plt.title(title)
                            plt.xlabel("Frame")
                            plt.ylabel("Angle Value")
                            plt.legend()
                            plt.tight_layout()
                            plt.savefig(plot_dir / f"{genotype}_trial{trial_idx}_{angle_col}_comparison.png")
                            plt.close()
                            
                    except Exception as e:
                        print(f"Error creating visualization for trial {trial_idx}, angle {angle_col}: {str(e)}")
                        traceback.print_exc()
    
    # Create summary statistics
    stats_df = pd.DataFrame(stats)
    
    # Print summary
    print("\nNormalization Summary:")
    summary = stats_df.groupby(['genotype', 'angle', 'dominant_sign']).size().reset_index(name='count')
    print(summary)
    
    # Print more detailed summary showing counts
    print("\nDetailed Sign Count Summary:")
    detailed_summary = stats_df.groupby(['genotype', 'angle']).agg({
        'positive_count': 'sum',
        'negative_count': 'sum',
        'zero_count': 'sum'
    }).reset_index()
    print(detailed_summary)
    
    # Save statistics
    stats_df.to_csv("angle_normalization_stats.csv", index=False)
    
    # Create summary plots - Plot separately for each genotype
    try:
        genotypes = stats_df['genotype'].unique()
        
        # Skip plotting if no data was collected
        if len(genotypes) == 0 or len(stats_df) == 0:
            print("No data collected for plotting. Skipping plot generation.")
        else:
            # Create a single figure with subplots for each genotype
            plt.figure(figsize=(15, 10))
            
            for i, genotype in enumerate(genotypes):
                try:
                    plt.subplot(len(genotypes), 1, i+1)
                    genotype_data = stats_df[stats_df['genotype'] == genotype]
                    
                    # Only plot if there's data for this genotype
                    if len(genotype_data) > 0:
                        sns.countplot(data=genotype_data, x='angle', hue='dominant_sign')
                        plt.title(f"Dominant Sign Distribution for {genotype}")
                        plt.xticks(rotation=45, ha='right')
                    else:
                        plt.text(0.5, 0.5, f"No data for {genotype}", 
                                horizontalalignment='center', verticalalignment='center')
                except Exception as e:
                    print(f"Error plotting for genotype {genotype}: {str(e)}")
                    plt.text(0.5, 0.5, f"Error plotting {genotype}: {str(e)}", 
                             horizontalalignment='center', verticalalignment='center')
            
            plt.tight_layout()
            plt.savefig(plot_dir / "dominant_sign_counts.png")
            plt.close()
    except Exception as e:
        print(f"Error generating summary plots: {str(e)}")
    
    return normalized_data

def verify_normalization(data, normalized_data, angle_cols):
    """
    Verify that normalization worked as expected.
    
    Args:
        data: Original DataFrame
        normalized_data: Normalized DataFrame
        angle_cols: List of angle columns that were normalized
    """
    print("\nVerifying normalization results:")
    
    # Create summary table
    summary_data = []
    
    # Check for each angle column
    for angle_col in angle_cols:
        # Calculate how many values changed sign
        original_values = data[angle_col].dropna().values
        normalized_values = normalized_data[angle_col].dropna().values
        
        original_signs = np.sign(original_values)
        normalized_signs = np.sign(normalized_values)
        different_signs = np.sum(original_signs != normalized_signs)
        
        # Count positive, negative, and zero values before normalization
        pos_values_before = np.sum(original_signs > 0)
        neg_values_before = np.sum(original_signs < 0)
        zero_values_before = np.sum(original_signs == 0)
        
        # Count positive, negative, and zero values after normalization
        pos_values_after = np.sum(normalized_signs > 0)
        neg_values_after = np.sum(normalized_signs < 0)
        zero_values_after = np.sum(normalized_signs == 0)
        
        # Calculate consistency - all should be positive OR all should be negative
        is_consistent = (pos_values_after == 0 or neg_values_after == 0)
        
        # Determine dominant sign
        dominant_sign_label = "positive" if pos_values_before > neg_values_before else "negative"
        expected_pos = len(original_signs) if dominant_sign_label == "positive" else 0
        expected_neg = len(original_signs) if dominant_sign_label == "negative" else 0
        
        # Calculate flipped percentages
        if len(original_signs) > 0:
            pct_flipped = different_signs / len(original_signs) * 100
            pct_pos_before = pos_values_before / len(original_signs) * 100
            pct_neg_before = neg_values_before / len(original_signs) * 100
            pct_pos_after = pos_values_after / len(original_signs) * 100
            pct_neg_after = neg_values_after / len(normalized_signs) * 100
        else:
            pct_flipped = pct_pos_before = pct_neg_before = pct_pos_after = pct_neg_after = 0
        
        # Add to summary data
        summary_data.append({
            'angle': angle_col,
            'total_values': len(original_signs),
            'values_flipped': different_signs,
            'pct_flipped': pct_flipped,
            'pos_before': pos_values_before,
            'pct_pos_before': pct_pos_before,
            'neg_before': neg_values_before,
            'pct_neg_before': pct_neg_before,
            'pos_after': pos_values_after,
            'pct_pos_after': pct_pos_after,
            'neg_after': neg_values_after,
            'pct_neg_after': pct_neg_after,
            'dominant_sign': dominant_sign_label,
            'is_consistent': is_consistent
        })
        
        # Print statistics for each angle
        print(f"\n{angle_col}:")
        print(f"  Total values: {len(original_signs)}")
        print(f"  Dominant sign: {dominant_sign_label.upper()}")
        print(f"  Values with changed signs: {different_signs} ({pct_flipped:.2f}%)")
        print(f"  Positive values before: {pos_values_before} ({pct_pos_before:.2f}%)")
        print(f"  Negative values before: {neg_values_before} ({pct_neg_before:.2f}%)")
        print(f"  Positive values after: {pos_values_after} ({pct_pos_after:.2f}%)")
        print(f"  Negative values after: {neg_values_after} ({pct_neg_after:.2f}%)")
        
        if not is_consistent:
            print(f"  WARNING: Values are not completely consistent after normalization!")
            print(f"  Expected all {dominant_sign_label} values, but got {pos_values_after} positive and {neg_values_after} negative")
    
    # Create and print summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Calculate overall statistics
    total_values = summary_df['total_values'].sum()
    total_flipped = summary_df['values_flipped'].sum()
    pct_total_flipped = (total_flipped / total_values * 100) if total_values > 0 else 0
    
    total_pos_before = summary_df['pos_before'].sum()
    total_neg_before = summary_df['neg_before'].sum()
    pct_pos_before = (total_pos_before / total_values * 100) if total_values > 0 else 0
    pct_neg_before = (total_neg_before / total_values * 100) if total_values > 0 else 0
    
    total_pos_after = summary_df['pos_after'].sum()
    total_neg_after = summary_df['neg_after'].sum()
    pct_pos_after = (total_pos_after / total_values * 100) if total_values > 0 else 0
    pct_neg_after = (total_neg_after / total_values * 100) if total_values > 0 else 0
    
    # Count inconsistencies
    inconsistent_angles = summary_df[summary_df['is_consistent'] == False]['angle'].tolist()
    
    print("\nOverall Normalization Summary:")
    print(f"  Total angles processed: {len(angle_cols)}")
    print(f"  Total values processed: {total_values}")
    print(f"  Total values flipped: {total_flipped} ({pct_total_flipped:.2f}%)")
    print(f"  Positive values before: {total_pos_before} ({pct_pos_before:.2f}%)")
    print(f"  Negative values before: {total_neg_before} ({pct_neg_before:.2f}%)")
    print(f"  Positive values after: {total_pos_after} ({pct_pos_after:.2f}%)")
    print(f"  Negative values after: {total_neg_after} ({pct_neg_after:.2f}%)")
    
    if inconsistent_angles:
        print(f"\nWARNING: Found {len(inconsistent_angles)} angles with inconsistent signs after normalization:")
        for angle in inconsistent_angles:
            print(f"  - {angle}")
    else:
        print(f"\nAll angles have been successfully normalized to their dominant sign.")
    
    # Save the verification summary
    summary_df.to_csv("normalization_verification.csv", index=False)
    print(f"\nDetailed verification summary saved to normalization_verification.csv")

def main():
    """Main function to normalize angle data across datasets."""
    try:
        # Process normal fly dataset (BPN, P9RT, P9LT)
        print("\nProcessing BPN, P9RT, P9LT dataset:")
        normal_data_path = get_available_data_path('BPN')
        
        if normal_data_path.endswith('.csv'):
            normal_data = pd.read_csv(normal_data_path)
        elif normal_data_path.endswith('.parquet'):
            normal_data = pd.read_parquet(normal_data_path)
        else:
            raise ValueError(f"Unsupported file format: {normal_data_path}")
        
        # Reset index to ensure we have unique, integer-based indices
        normal_data = normal_data.reset_index(drop=True)
        
        print(f"Loaded data with shape: {normal_data.shape}")
        print(f"Original columns: {len(normal_data.columns)}")
        
        # Get all abduct and rot angle columns
        abduct_cols = [col for col in normal_data.columns if col.endswith('_abduct')]
        rot_cols = [col for col in normal_data.columns if col.endswith('_rot')]
        angle_cols = abduct_cols + rot_cols
        non_angle_cols = [col for col in normal_data.columns if col not in angle_cols]
        
        print(f"Found {len(abduct_cols)} abduct columns and {len(rot_cols)} rot columns")
        print(f"Abduct columns: {abduct_cols}")
        print(f"Rot columns: {rot_cols}")
        print(f"Found {len(non_angle_cols)} non-angle columns that will be preserved")
        
        # Save a backup of the original data for verification
        original_normal_data = normal_data.copy()
        
        # Normalize angles for BPN, P9RT, P9LT dataset
        # Note: frame_range is now only used for visualization highlighting
        normal_data_normalized = normalize_angles(
            normal_data, 
            trial_size=1400, 
            angle_cols=angle_cols,
            frame_range=(400, 1000)  # Only used for visualization highlighting
        )
        
        # Verify normalization was successful
        verify_normalization(original_normal_data, normal_data_normalized, angle_cols)
        
        # Create output file path with timestamp to avoid overwriting
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        file_basename = Path(normal_data_path).stem
        file_extension = Path(normal_data_path).suffix
        output_dir = Path("normalized_data")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        output_path = output_dir / f"{file_basename}_normalized_{timestamp}{file_extension}"
        
        # Save normalized data
        if file_extension.lower() == '.csv':
            normal_data_normalized.to_csv(output_path, index=False)
        elif file_extension.lower() == '.parquet':
            normal_data_normalized.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        print(f"Saved normalized data to {output_path}")
        print(f"Output data shape: {normal_data_normalized.shape}")
        print(f"All columns from the original dataset are preserved")
        
        # Compare a few values before and after normalization
        print("\nComparing a few values before and after normalization:")
        for angle_col in angle_cols[:3]:  # Show first three angle columns
            original_values = original_normal_data[angle_col].iloc[:5].values  # First 5 values
            normalized_values = normal_data_normalized[angle_col].iloc[:5].values
            print(f"\n{angle_col}:")
            print(f"  Original values: {original_values}")
            print(f"  Normalized values: {normalized_values}")
        
        # Process ES dataset if available
        try:
            print("\nProcessing ES dataset:")
            es_data_path = get_available_data_path('ES')
            
            if es_data_path.endswith('.csv'):
                es_data = pd.read_csv(es_data_path)
            elif es_data_path.endswith('.parquet'):
                es_data = pd.read_parquet(es_data_path)
            else:
                raise ValueError(f"Unsupported file format: {es_data_path}")
            
            # Reset index to ensure we have unique, integer-based indices
            es_data = es_data.reset_index(drop=True)
            
            print(f"Loaded ES data with shape: {es_data.shape}")
            print(f"Original columns: {len(es_data.columns)}")
            
            # Get all abduct and rot angle columns for ES
            es_abduct_cols = [col for col in es_data.columns if col.endswith('_abduct')]
            es_rot_cols = [col for col in es_data.columns if col.endswith('_rot')]
            es_angle_cols = es_abduct_cols + es_rot_cols
            es_non_angle_cols = [col for col in es_data.columns if col not in es_angle_cols]
            
            print(f"Found {len(es_abduct_cols)} abduct columns and {len(es_rot_cols)} rot columns in ES data")
            print(f"Found {len(es_non_angle_cols)} non-angle columns that will be preserved")
            
            # Save a backup of the original ES data for verification
            original_es_data = es_data.copy()
            
            # Normalize angles for ES dataset
            # Note: frame_range is now only used for visualization highlighting
            es_data_normalized = normalize_angles(
                es_data, 
                trial_size=1400,  # Adjust if ES trial size is different
                angle_cols=es_angle_cols,
                frame_range=(0, 650)  # Only used for visualization highlighting
            )
            
            # Verify normalization was successful
            verify_normalization(original_es_data, es_data_normalized, es_angle_cols)
            
            # Create output file path with timestamp
            es_file_basename = Path(es_data_path).stem
            es_file_extension = Path(es_data_path).suffix
            es_output_path = output_dir / f"{es_file_basename}_normalized_{timestamp}{es_file_extension}"
            
            # Save normalized ES data
            if es_file_extension.lower() == '.csv':
                es_data_normalized.to_csv(es_output_path, index=False)
            elif es_file_extension.lower() == '.parquet':
                es_data_normalized.to_parquet(es_output_path, index=False)
            else:
                raise ValueError(f"Unsupported file format: {es_file_extension}")
            
            print(f"Saved normalized ES data to {es_output_path}")
            print(f"Output data shape: {es_data_normalized.shape}")
            print(f"All columns from the original ES dataset are preserved")
            
            # Compare a few values before and after normalization
            print("\nComparing a few values before and after normalization for ES data:")
            for angle_col in es_angle_cols[:3]:  # Show first three angle columns
                original_values = original_es_data[angle_col].iloc[:5].values  # First 5 values
                normalized_values = es_data_normalized[angle_col].iloc[:5].values
                print(f"\n{angle_col}:")
                print(f"  Original values: {original_values}")
                print(f"  Normalized values: {normalized_values}")
            
        except FileNotFoundError:
            print("ES dataset not found, skipping")
        except Exception as e:
            print(f"Error processing ES dataset: {str(e)}")
            traceback.print_exc()
    
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 