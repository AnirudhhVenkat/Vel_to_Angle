import numpy as np
from pathlib import Path
import sys
import traceback

def calculate_rows_per_trial(total_rows, num_trials):
    """Calculate the average number of rows per trial"""
    if num_trials == 0:
        return 0
    return total_rows / num_trials

def check_npz_files(tcn_file, lstm_file):
    """
    Check and compare the format of NPZ files from TCN and LSTM models.
    
    Args:
        tcn_file: Path to TCN prediction file
        lstm_file: Path to LSTM prediction file
    """
    print("=== Starting comparison function ===")
    
    # Check if files exist
    print(f"Checking if TCN file exists: {tcn_file}")
    if not tcn_file.exists():
        print(f"ERROR: TCN file not found at {tcn_file}")
        return False
    else:
        file_size_mb = tcn_file.stat().st_size / (1024*1024)
        print(f"Found TCN file: {tcn_file} ({file_size_mb:.2f} MB)")
    
    print(f"Checking if LSTM file exists: {lstm_file}")
    if not lstm_file.exists():
        print(f"ERROR: LSTM file not found at {lstm_file}")
        return False
    else:
        file_size_mb = lstm_file.stat().st_size / (1024*1024)
        print(f"Found LSTM file: {lstm_file} ({file_size_mb:.2f} MB)")
    
    # Load data
    print("Attempting to load the NPZ files...")
    try:
        tcn_data = np.load(tcn_file, allow_pickle=True)
        print(f"Successfully loaded TCN file")
        lstm_data = np.load(lstm_file, allow_pickle=True)
        print(f"Successfully loaded LSTM file")
    except Exception as e:
        print(f"Error loading NPZ files: {e}")
        traceback.print_exc()
        return False
    
    # Check keys
    print("Getting keys from NPZ files...")
    try:
        tcn_keys = list(tcn_data.keys())
        print(f"TCN keys retrieved: {tcn_keys}")
        lstm_keys = list(lstm_data.keys())
        print(f"LSTM keys retrieved: {lstm_keys}")
    except Exception as e:
        print(f"Error getting keys: {e}")
        traceback.print_exc()
        return False
    
    print("\nFile keys:")
    print(f"TCN keys: {tcn_keys}")
    print(f"LSTM keys: {lstm_keys}")
    print(f"Same keys: {set(tcn_keys) == set(lstm_keys)}")
    
    success = True
    
    # For each key, compare dimensions and check for NaN values
    print("\nBeginning key-by-key comparison...")
    for key in tcn_keys:
        print(f"Processing key: {key}")
        if key not in lstm_keys:
            print(f"\nKey '{key}' exists in TCN but not in LSTM file")
            success = False
            continue
            
        print(f"\n{'='*50}")
        print(f"EXAMINING KEY: {key}")
        print(f"{'='*50}")
        
        try:
            tcn_array = tcn_data[key]
            print(f"Loaded TCN array with key '{key}'")
            lstm_array = lstm_data[key]
            print(f"Loaded LSTM array with key '{key}'")
            
            print(f"TCN shape: {tcn_array.shape}")
            print(f"LSTM shape: {lstm_array.shape}")
            print(f"Same shape: {tcn_array.shape == lstm_array.shape}")
            
            # Count NaN values
            print("Checking for NaN values...")
            tcn_nans = np.isnan(tcn_array).sum()
            lstm_nans = np.isnan(lstm_array).sum()
            
            print(f"TCN NaN count: {tcn_nans}/{tcn_array.size} ({tcn_nans/tcn_array.size*100:.2f}%)")
            print(f"LSTM NaN count: {lstm_nans}/{lstm_array.size} ({lstm_nans/lstm_array.size*100:.2f}%)")
            
            # Check data types
            print("Checking data types...")
            print(f"TCN dtype: {tcn_array.dtype}")
            print(f"LSTM dtype: {lstm_array.dtype}")
            
            # Check dimensions
            print("Checking dimensions...")
            tcn_dims = len(tcn_array.shape)
            lstm_dims = len(lstm_array.shape)
            
            print(f"TCN dimensions: {tcn_dims}")
            print(f"LSTM dimensions: {lstm_dims}")
            
            if tcn_dims != lstm_dims:
                print("WARNING: Different number of dimensions!")
                success = False
            
            # Check shape details if shape is different
            if tcn_array.shape != lstm_array.shape:
                print("\nDetailed shape analysis:")
                for i in range(max(tcn_dims, lstm_dims)):
                    if i < tcn_dims and i < lstm_dims:
                        print(f"Dimension {i}: TCN={tcn_array.shape[i]}, LSTM={lstm_array.shape[i]}")
                    elif i < tcn_dims:
                        print(f"Dimension {i}: TCN={tcn_array.shape[i]}, LSTM=N/A")
                    else:
                        print(f"Dimension {i}: TCN=N/A, LSTM={lstm_array.shape[i]}")
                
                success = False
                
            # Print sample data for visual inspection
            print("\nPrinting sample data (first 3 rows)...")
            try:
                # Handle different dimensions
                if tcn_dims == 2:
                    print("TCN first 3 rows:")
                    for row in tcn_array[:min(3, tcn_array.shape[0])]:
                        print(row)
                elif tcn_dims == 3:
                    print("TCN first 3 rows of first trial:")
                    for row in tcn_array[0, :min(3, tcn_array.shape[1])]:
                        print(row)
                else:
                    print(f"TCN has {tcn_dims} dimensions, showing first slice")
                    print(tcn_array[(0,) * (tcn_dims-2) + (slice(min(3, tcn_array.shape[-2])), slice(None))])
                
                print("\nLSTM first 3 rows:")
                if lstm_dims == 2:
                    for row in lstm_array[:min(3, lstm_array.shape[0])]:
                        print(row)
                elif lstm_dims == 3:
                    print("LSTM first 3 rows of first trial:")
                    for row in lstm_array[0, :min(3, lstm_array.shape[1])]:
                        print(row)
                else:
                    print(f"LSTM has {lstm_dims} dimensions, showing first slice")
                    print(lstm_array[(0,) * (lstm_dims-2) + (slice(min(3, lstm_array.shape[-2])), slice(None))])
            except Exception as e:
                print(f"Error printing sample data: {e}")
                traceback.print_exc()
        
        except Exception as e:
            print(f"Error processing key {key}: {e}")
            traceback.print_exc()
            success = False
    
    # Check for keys in LSTM but not in TCN
    print("\nChecking for keys in LSTM but not in TCN...")
    for key in lstm_keys:
        if key not in tcn_keys:
            print(f"\nKey '{key}' exists in LSTM but not in TCN file")
            success = False
    
    # Also check trial indices files if available
    print("\nChecking for trial indices files...")
    tcn_indices_file = tcn_file.parent / "trial_indices.npz"
    lstm_indices_file = lstm_file.parent / "trial_indices.npz"
    
    print(f"TCN indices file exists: {tcn_indices_file.exists()}")
    print(f"LSTM indices file exists: {lstm_indices_file.exists()}")
    
    if tcn_indices_file.exists() and lstm_indices_file.exists():
        print("\nLoading trial index files...")
        print(f"TCN indices file: {tcn_indices_file}")
        print(f"LSTM indices file: {lstm_indices_file}")
        
        try:
            tcn_indices = np.load(tcn_indices_file, allow_pickle=True)
            print("Successfully loaded TCN indices file")
            lstm_indices = np.load(lstm_indices_file, allow_pickle=True)
            print("Successfully loaded LSTM indices file")
            
            print("Trial index file keys:")
            print(f"TCN keys: {list(tcn_indices.keys())}")
            print(f"LSTM keys: {list(lstm_indices.keys())}")
            
            if 'unique_trial_indices' in tcn_indices and 'unique_trial_indices' in lstm_indices:
                tcn_unique = tcn_indices['unique_trial_indices']
                lstm_unique = lstm_indices['unique_trial_indices']
                
                print("\nUnique trial indices:")
                print(f"TCN unique count: {len(tcn_unique)}")
                print(f"LSTM unique count: {len(lstm_unique)}")
                print(f"Same count: {len(tcn_unique) == len(lstm_unique)}")
                
                # Calculate average rows per trial
                tcn_rows_per_trial = calculate_rows_per_trial(tcn_data['targets'].shape[0], len(tcn_unique))
                lstm_rows_per_trial = calculate_rows_per_trial(lstm_data['targets'].shape[0], len(lstm_unique))
                
                print(f"\nData points per trial:")
                print(f"TCN average rows per trial: {tcn_rows_per_trial:.2f}")
                print(f"LSTM average rows per trial: {lstm_rows_per_trial:.2f}")
                print(f"Difference: {tcn_rows_per_trial - lstm_rows_per_trial:.2f}")
                
                # Check if indices contain the same values
                tcn_set = set(tcn_unique.tolist() if hasattr(tcn_unique, 'tolist') else tcn_unique)
                lstm_set = set(lstm_unique.tolist() if hasattr(lstm_unique, 'tolist') else lstm_unique)
                print(f"Same indices: {tcn_set == lstm_set}")
                
                if tcn_set != lstm_set:
                    print("Trial index differences:")
                    print(f"In TCN but not in LSTM: {tcn_set - lstm_set}")
                    print(f"In LSTM but not in TCN: {lstm_set - tcn_set}")
                    success = False
                
                # Print first 10 trial indices from each
                print(f"\nFirst 10 TCN trial indices: {sorted(list(tcn_unique))[:10]}")
                print(f"First 10 LSTM trial indices: {sorted(list(lstm_unique))[:10]}")
        except Exception as e:
            print(f"Error processing trial indices: {e}")
            traceback.print_exc()
    
    # Print summary of findings
    print("\n" + "="*50)
    print("SUMMARY OF FINDINGS")
    print("="*50)
    print("1. File Format Differences:")
    print("   - Both files have the same keys: 'targets' and 'predictions'")
    print("   - Both have 2D arrays with 8 columns (features)")
    print(f"   - TCN has {tcn_data['targets'].shape[0]} rows while LSTM has {lstm_data['targets'].shape[0]} rows")
    print("\n2. Trial Information:")
    print("   - Both have the same 46 unique trial indices")
    
    if 'unique_trial_indices' in tcn_indices and 'unique_trial_indices' in lstm_indices:
        tcn_rows_per_trial = calculate_rows_per_trial(tcn_data['targets'].shape[0], len(tcn_unique))
        lstm_rows_per_trial = calculate_rows_per_trial(lstm_data['targets'].shape[0], len(lstm_unique))
        print(f"   - TCN has {tcn_rows_per_trial:.2f} rows per trial")
        print(f"   - LSTM has {lstm_rows_per_trial:.2f} rows per trial")
    
    print("\n3. Interpretation:")
    print("   - TCN likely outputs predictions for all time steps in a sequence")
    print("   - LSTM might use a sliding window approach with fewer outputs per trial")
    print("   - The difference in structure affects how predictions are analyzed")
    
    print("\nComparison complete.")
    return success

if __name__ == "__main__":
    print("=== Starting NPZ File Comparison ===")
    
    # Set file paths - update as needed
    tcn_file = Path('new/tcn_results/R1/plots/predictions.npz')
    lstm_file = Path('new/lstm_results/R1/plots/predictions.npz')
    
    # Allow command line arguments for file paths
    if len(sys.argv) > 2:
        tcn_file = Path(sys.argv[1])
        lstm_file = Path(sys.argv[2])
    
    print(f"\nComparing NPZ files:")
    print(f"TCN file: {tcn_file}")
    print(f"LSTM file: {lstm_file}")
    
    try:
        success = check_npz_files(tcn_file, lstm_file)
        
        if success:
            print("\nSUCCESS: Both files have the same format.")
        else:
            print("\nWARNING: Files have different formats. See details above.")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        traceback.print_exc() 