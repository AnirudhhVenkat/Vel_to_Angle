import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import itertools
from sklearn.model_selection import KFold
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import os
import argparse

def check_gpu_availability():
    """Check if GPU is available and can be used by XGBoost."""
    try:
        # Create a small dummy dataset
        X = np.random.normal(size=(100, 10))
        y = np.random.normal(size=100)
        
        # Try to train a small model with GPU - using new XGBoost 2.0+ syntax
        model = xgb.XGBRegressor(tree_method='hist', device='cuda')
        model.fit(X, y)
        
        # If we get here, GPU is working
        print("\nGPU is available and working with XGBoost!")
        print("Using CUDA device for training")
        return True
        
    except Exception as e:
        print("\nGPU acceleration is not available:")
        print(f"Error: {str(e)}")
        print("\nFalling back to CPU. To enable GPU acceleration, please:")
        print("1. Install CUDA toolkit")
        print("2. Make sure your GPU is CUDA-compatible")
        return False

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
        
        # Create trial IDs based on frame numbers
        frame_nums = df['fnum'].values
        trial_ids = []
        current_trial = 0
        last_frame = frame_nums[0]
        
        for frame in frame_nums:
            if frame < last_frame:  # New trial starts when frame number resets
                current_trial += 1
            trial_ids.append(current_trial)
            last_frame = frame
        
        df['trial_id'] = trial_ids
        print(f"Initial number of trials: {len(df['trial_id'].unique())}")
        print(f"Frames per trial: {len(df) / len(df['trial_id'].unique()):.1f}")
        
        # Verify trial lengths and keep only complete trials
        trial_lengths = df.groupby('trial_id').size()
        print("\nTrial length statistics before filtering:")
        print(trial_lengths.describe())
        
        # Keep only trials with exactly 1400 frames
        complete_trials = trial_lengths[trial_lengths == 1400].index
        df = df[df['trial_id'].isin(complete_trials)].copy()
        print(f"\nKeeping only complete 1400-frame trials:")
        print(f"Remaining trials: {len(complete_trials)}")
        print(f"Total frames: {len(df)}")
        
        # Filter frames 400-1000 from each trial
        filtered_rows = []
        for trial in df['trial_id'].unique():
            trial_data = df[df['trial_id'] == trial]
            filtered_rows.append(trial_data.iloc[400:1000])
        
        df = pd.concat(filtered_rows, axis=0, ignore_index=True)
        print(f"\nFiltered to frames 400-1000:")
        print(f"Number of trials: {len(filtered_rows)}")
        print(f"Total frames: {len(df)} ({len(df) / len(filtered_rows):.1f} frames per trial)")
        
        return df
        
    except Exception as e:
        print(f"Error loading/preprocessing data: {e}")
        raise

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

def prepare_data_for_xgboost(genotype, window_size=100):
    """Prepare data for XGBoost training."""
    print("\nPreparing data...")
    
    # Load data
    file_path = get_available_data_path()
    df = pd.read_csv(file_path)
    print(f"Loaded data with shape: {df.shape}")
    
    # Filter for specific genotype if requested
    if genotype != 'ALL':
        df = df[df['genotype'] == genotype].copy()
        print(f"Filtered for {genotype} genotype: {df.shape}")
    
    # Create trial IDs based on frame numbers
    trial_size = 1400
    num_trials = len(df) // trial_size
    print(f"\nInitial number of trials: {num_trials}")
    print(f"Total frames: {len(df)}")
    print(f"Remainder frames: {len(df) % trial_size}")
    
    # Keep only complete trials
    complete_trials_data = df.iloc[:num_trials * trial_size].copy()
    print(f"Keeping only complete trials: {len(complete_trials_data)} frames")
    
    # Create trial IDs
    complete_trials_data['trial_id'] = np.repeat(np.arange(num_trials), trial_size)
    
    # Calculate split sizes
    train_size = int(0.7 * num_trials)
    val_size = int(0.15 * num_trials)
    test_size = num_trials - train_size - val_size
    
    print(f"\nSplitting data by trials:")
    print(f"Train: {train_size} trials")
    print(f"Validation: {val_size} trials")
    print(f"Test: {test_size} trials")
    
    # Create random permutation of trial indices
    np.random.seed(42)  # For reproducibility
    trial_indices = np.random.permutation(num_trials)
    
    # Split trial indices into train/val/test
    train_trials = trial_indices[:train_size]
    val_trials = trial_indices[train_size:train_size + val_size]
    test_trials = trial_indices[train_size + val_size:]
    
    print("\nTrial assignments:")
    print(f"Training trials: {sorted(train_trials)}")
    print(f"Validation trials: {sorted(val_trials)}")
    print(f"Test trials: {sorted(test_trials)}")
    
    # Create masks for each split
    train_mask = np.zeros(len(complete_trials_data), dtype=bool)
    val_mask = np.zeros(len(complete_trials_data), dtype=bool)
    test_mask = np.zeros(len(complete_trials_data), dtype=bool)
    
    # Assign trials to splits using the random indices
    for trial in train_trials:
        start_idx = trial * trial_size
        end_idx = (trial + 1) * trial_size
        train_mask[start_idx:end_idx] = True
    
    for trial in val_trials:
        start_idx = trial * trial_size
        end_idx = (trial + 1) * trial_size
        val_mask[start_idx:end_idx] = True
    
    for trial in test_trials:
        start_idx = trial * trial_size
        end_idx = (trial + 1) * trial_size
        test_mask[start_idx:end_idx] = True
    
    # Split data
    train_data = complete_trials_data[train_mask].copy()
    val_data = complete_trials_data[val_mask].copy()
    test_data = complete_trials_data[test_mask].copy()
    
    return train_data, val_data, test_data

def prepare_features_and_targets(train_data, val_data, test_data, window_size=100):
    """Prepare features and targets for XGBoost training."""
    
    # Calculate moving averages for velocities within each split
    base_velocities = ['x_vel', 'y_vel', 'z_vel']
    for data in [train_data, val_data, test_data]:
        for window in [5, 10, 20]:
            for vel in base_velocities:
                data[f'{vel}_ma{window}'] = data[vel].rolling(window=window, center=True, min_periods=1).mean()
    
    # Define features and targets
    features = [
        'R-F-CTr_y',    # Starting point for prediction chain
        'x_vel_ma5', 'y_vel_ma5', 'z_vel_ma5',
        'x_vel_ma10', 'y_vel_ma10', 'z_vel_ma10',
        'x_vel_ma20', 'y_vel_ma20', 'z_vel_ma20',
        'x_vel', 'y_vel', 'z_vel'
    ]
    
    targets = [
        'R1A_flex', 'R1A_rot', 'R1A_abduct',
        'R1B_flex', 'R1B_rot',
        'R1C_flex', 'R1C_rot',
        'R1D_flex'
    ]
    
    # Filter frames 400-1000 for prediction
    context_start = 400 - window_size
    train_data = train_data.iloc[context_start:1000].copy()
    val_data = val_data.iloc[context_start:1000].copy()
    test_data = test_data.iloc[context_start:1000].copy()
    
    print("\nFeature and Target Information:")
    print(f"Features ({len(features)}):")
    for feature in features:
        if feature in train_data.columns:
            print(f"  ✓ {feature}")
            print(f"    NaN count: {train_data[feature].isna().sum()}")
        else:
            print(f"  ✗ {feature} (not found)")
    
    print(f"\nTargets ({len(targets)}):")
    for target in targets:
        if target in train_data.columns:
            print(f"  ✓ {target}")
            print(f"    NaN count: {train_data[target].isna().sum()}")
        else:
            print(f"  ✗ {target} (not found)")
    
    # Prepare feature matrices and target vectors
    X_train = train_data[features]
    X_val = val_data[features]
    X_test = test_data[features]
    
    y_train = train_data[targets]
    y_val = val_data[targets]
    y_test = test_data[targets]
    
    print("\nDataset Shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"X_test: {X_test.shape}")
    
    return (X_train, X_val, X_test), (y_train, y_val, y_test), features, targets

def find_optimal_params(X_train, X_val, y_train, y_val, target_name, parent_dir, genotype, use_gpu=False):
    """Find optimal hyperparameters using validation set."""
    print(f"\nFinding optimal parameters for {target_name}...")
    
    # Base parameters for grid search - focused around best performing values
    base_params = {
        # Tree structure parameters
        'max_depth': [4, 5, 6],              # Center around 5
        'min_child_weight': [3],       # Center around 3
        
        # Sampling parameters
        'subsample': [0.8],        # Center around 0.8
        'colsample_bytree': [0.8], # Center around 0.8
        
        # Learning parameters
        'learning_rate': [0.005, 0.01, 0.02], # Focus on lower learning rates
        'n_estimators': [2000, 3000],   # Number of trees
        
        # Fixed parameters
        'tree_method': ['hist'],
        'objective': ['reg:absoluteerror'],  # Use MAE loss
        'eval_metric': ['mae'],    # Evaluate using MAE
        'early_stopping_rounds': [10],      # Early stopping
        'verbosity': [0]                    # Less verbose output
    }
    
    if use_gpu:
        base_params['tree_method'] = ['gpu_hist']
        base_params['device'] = ['cuda']
    
    # Create parameter combinations
    param_combinations = [dict(zip(base_params.keys(), v)) 
                        for v in itertools.product(*base_params.values())]
    
    print(f"Total parameter combinations to try: {len(param_combinations)}")
    
    param_search_results = []
    
    for params in tqdm(param_combinations, desc="Parameter combinations"):
        # Extract early stopping rounds
        early_stopping = params.pop('early_stopping_rounds')
        
        # Create and train model
        model = XGBRegressor(**params)
        
        # Train with early stopping using validation set
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate on validation set
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        param_search_results.append({
            'params': params,
            'mae': mae,
            'r2': r2
        })
    
    # Find best parameters based on MAE
    results_df = pd.DataFrame(param_search_results)
    results_df = results_df.sort_values('mae')
    
    best_params = results_df.iloc[0]['params']
    best_mae = results_df.iloc[0]['mae']
    best_r2 = results_df.iloc[0]['r2']
    
    print(f"\nBest parameters for {target_name}:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best MAE: {best_mae:.4f}")
    print(f"Best R²: {best_r2:.4f}")
    
    return best_params

def train_xgboost_model(X_train, X_val, y_train, y_val, params, target_name, parent_dir, genotype, use_gpu=False):
    """Train an XGBoost model with given parameters."""
    print(f"\nTraining XGBoost model for {target_name}...")
    
    # Ensure MAE loss is set
    final_params = params.copy()
    final_params['objective'] = 'reg:absoluteerror'
    final_params['eval_metric'] = 'mae'
    
    # Create model with parameters
    model = XGBRegressor(**final_params)
    if use_gpu:
        model.set_params(tree_method='gpu_hist', device='cuda')
    
    # Create output directory
    output_dir = parent_dir / target_name
    output_dir.mkdir(exist_ok=True)
    
    # Train the model
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    # Make predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # Calculate metrics
    train_metrics = {
        'r2': r2_score(y_train, train_pred),
        'mae': mean_absolute_error(y_train, train_pred),
        'mse': mean_squared_error(y_train, train_pred)
    }
    
    val_metrics = {
        'r2': r2_score(y_val, val_pred),
        'mae': mean_absolute_error(y_val, val_pred),
        'mse': mean_squared_error(y_val, val_pred)
    }
    
    # Save metrics
    with open(output_dir / 'metrics.txt', 'w') as f:
        f.write("Training Metrics:\n")
        for metric, value in train_metrics.items():
            f.write(f"{metric.upper()}: {value:.4f}\n")
        f.write("\nValidation Metrics:\n")
        for metric, value in val_metrics.items():
            f.write(f"{metric.upper()}: {value:.4f}\n")
        
        # Save feature importances
        f.write("\nFeature Importances:\n")
        importances = dict(zip(X_train.columns, model.feature_importances_))
        for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{feat}: {imp:.4f}\n")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 5))
    
    # Training set
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, train_pred, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Training Set\nR² = {train_metrics["r2"]:.3f}')
    
    # Validation set
    plt.subplot(1, 2, 2)
    plt.scatter(y_val, val_pred, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Validation Set\nR² = {val_metrics["r2"]:.3f}')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'actual_vs_predicted.png')
    plt.close()
    
    # Save model
    model.save_model(str(output_dir / f'{target_name}_model.json'))
    
    print(f"\nModel training completed for {target_name}")
    print(f"Training R²: {train_metrics['r2']:.4f}")
    print(f"Validation R²: {val_metrics['r2']:.4f}")
    
    return model

def train_xgboost_models(X_train, X_val, y_train, y_val, features, targets, genotype, parent_dir, use_gpu=False):
    """Train separate XGBoost models for each target variable with optimized parameters."""
    models = {}
    
    for target in targets:
        print(f"\nTraining model for {target}...")
        
        # Find optimal parameters using validation set
        best_params = find_optimal_params(X_train, X_val, y_train[target], y_val[target], 
                                        target, parent_dir, genotype, use_gpu)
        
        # Train final model with best parameters
        final_params = best_params.copy()
        final_params['n_estimators'] = 5000  # More trees for final model
        final_params['objective'] = 'reg:absoluteerror'  # Ensure MAE objective
        final_params['eval_metric'] = 'mae'  # Set eval metric
        
        # Add GPU device if available
        if use_gpu:
            final_params['device'] = 'cuda'
        
        model = train_xgboost_model(
            X_train, X_val, y_train[target], y_val[target],
            final_params, target, parent_dir, genotype, use_gpu
        )
        
        models[target] = model
        
        # Feature importance plot
        plt.figure(figsize=(8, 4))
        xgb.plot_importance(model)
        plt.title(f'Feature Importance for {target} ({genotype})')
        plt.tight_layout()
        
        # Save plot
        output_dir = parent_dir / genotype
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / f'feature_importance_{target}.png')
        plt.close()
        
        # Save best parameters and training info
        with open(output_dir / f'best_params_{target}.txt', 'w') as f:
            f.write(f"Best Parameters for {target}:\n")
            f.write("=" * (len(target) + 20) + "\n\n")
            f.write("Parameter Search Best Parameters:\n")
            for param, value in best_params.items():
                f.write(f"{param}: {value}\n")
            f.write("\nFinal Training Parameters:\n")
            for param, value in final_params.items():
                f.write(f"{param}: {value}\n")
            f.write(f"\nActual trees used: {model.get_booster().num_boosted_rounds()}\n")
    
    return models

def evaluate_models(models, X_test, y_test, features, targets, genotype, parent_dir):
    """Evaluate XGBoost models and generate predictions."""
    results = {}
    
    # Set style for better-looking plots
    plt.style.use('bmh')  # Using built-in style instead of seaborn
    
    for target in targets:
        model = models[target]
        predictions = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test[target], predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test[target], predictions)
        r2 = r2_score(y_test[target], predictions)
        
        results[target] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        # Calculate number of trials in test set
        trial_size = 600  # frames 400-1000
        num_trials = len(predictions) // trial_size
        print(f"\nNumber of test trials for {target}: {num_trials}")
        
        # Create plots directory
        output_dir = parent_dir / genotype / 'plots' / target
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Plot all trials in a scrolling plot
        plt.figure(figsize=(20, 6))
        time_points = np.arange(len(predictions))
        plt.plot(time_points, y_test[target], label='Actual', color='#2E86C1', alpha=0.8)
        plt.plot(time_points, predictions, label='Predicted', color='#E74C3C', alpha=0.8)
        
        # Add trial boundaries
        for i in range(num_trials):
            plt.axvline(x=i*trial_size, color='gray', linestyle='--', alpha=0.5)
        
        plt.title(f'All Trials Prediction vs Actual - {target} ({genotype})\nR² = {r2:.3f}, RMSE = {rmse:.3f}')
        plt.xlabel('Frame Number')
        plt.ylabel(f'{target} Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'all_trials_prediction.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Plot all trials in a multi-page figure
        trials_per_page = 4
        num_pages = (num_trials + trials_per_page - 1) // trials_per_page
        
        with PdfPages(output_dir / 'all_trials.pdf') as pdf:
            for page in range(num_pages):
                # Create figure with subplots for this page
                start_trial = page * trials_per_page
                end_trial = min((page + 1) * trials_per_page, num_trials)
                num_trials_this_page = end_trial - start_trial
                
                fig, axes = plt.subplots(num_trials_this_page, 1, figsize=(15, 5*num_trials_this_page))
                fig.suptitle(f'Trials {start_trial+1}-{end_trial} - {target} ({genotype})', fontsize=16)
                
                # Make axes iterable even if there's only one subplot
                if num_trials_this_page == 1:
                    axes = [axes]
                
                for trial_idx, trial in enumerate(range(start_trial, end_trial)):
                    ax = axes[trial_idx]
                    start_idx = trial * trial_size
                    end_idx = start_idx + trial_size
                    
                    # Get trial data
                    trial_actual = y_test[target].iloc[start_idx:end_idx]
                    trial_pred = predictions[start_idx:end_idx]
                    
                    # Calculate confidence band (using residuals)
                    residuals = trial_actual - trial_pred
                    std_residuals = np.std(residuals)
                    confidence_band = 1.96 * std_residuals  # 95% confidence interval
                    
                    # Plot actual vs predicted with confidence band
                    ax.plot(trial_actual.index, trial_actual.values, 'b-', label='Actual', alpha=0.7)
                    ax.plot(trial_actual.index, trial_pred, 'r-', label='Predicted', alpha=0.7)
                    ax.fill_between(trial_actual.index,
                                   trial_pred - confidence_band,
                                   trial_pred + confidence_band,
                                   color='gray', alpha=0.2,
                                   label='95% Confidence')
                    
                    # Calculate trial-specific metrics
                    trial_mae = mean_absolute_error(trial_actual, trial_pred)
                    trial_r2 = r2_score(trial_actual, trial_pred)
                    
                    ax.set_title(f'Trial {trial+1} (MAE: {trial_mae:.3f}, R²: {trial_r2:.3f})')
                    ax.set_xlabel('Frame')
                    ax.set_ylabel(f'{target} Value')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
        
        # 3. Plot residuals
        plt.figure(figsize=(10, 6))
        residuals = y_test[target] - predictions
        plt.scatter(predictions, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(f'Residual Plot - {target} ({genotype})')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'residuals.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Plot actual vs predicted scatter
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test[target], predictions, alpha=0.5)
        min_val = min(y_test[target].min(), predictions.min())
        max_val = max(y_test[target].max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.title(f'Actual vs Predicted - {target} ({genotype})')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print overall results
        print(f"\nResults for {target} ({genotype}):")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
    
    return results

def save_results(results, features, targets, genotype, parent_dir):
    """Save evaluation results to a file."""
    output_dir = parent_dir / genotype
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'model_results.txt', 'w') as f:
        f.write(f"XGBoost Model Results for {genotype}\n")
        f.write("=" * (24 + len(genotype)) + "\n\n")
        
        f.write("Features used:\n")
        for feature in features:
            f.write(f"- {feature}\n")
        f.write("\n")
        
        for target in targets:
            f.write(f"\nResults for {target}:\n")
            f.write("-" * (len(target) + 12) + "\n")
            metrics = results[target]
            for metric, value in metrics.items():
                f.write(f"{metric.upper()}: {value:.4f}\n")

def calculate_confidence_score(predictions, targets, target_name=None):
    """Calculate confidence score based on R² and MAE, weighted by position in chain."""
    r2 = r2_score(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    max_abs_target = np.max(np.abs(targets))
    normalized_mae = 1 - (mae / max_abs_target)
    
    # Adjust weights based on position in chain
    if target_name:
        # Weight R² more heavily for early chain predictions
        r2_weight = 0.8
        mae_weight = 0.2
    else:
        # Default weights if target_name not provided
        r2_weight = 0.8
        mae_weight = 0.2
    
    # Calculate confidence score
    confidence = r2_weight * r2 + mae_weight * normalized_mae
    
    return confidence

def train_prediction_chain(X_train, X_val, y_train, y_val, features, prediction_chain, chain_features, 
                         parent_dir, genotype, use_gpu=False):
    """Train a chain of XGBoost models starting with R-F-CTr_y to predict joint angles."""
    models = {}
    predictions = {}
    
    # Add debug prints
    print("\nInitial features in X_train:")
    print(X_train.columns.tolist())
    
    # Train model for each step in the chain (skip R-F-CTr_y as it's the input)
    for target in prediction_chain[1:]:
        print(f"\nTraining model for {target}...")
        
        # Get features for this target
        current_features = chain_features[target]
        
        # For subsequent predictions, we need to use the previous predictions
        if target != prediction_chain[1]:  # Not the first prediction
            print(f"\nBefore adding predictions for {target}, X_train features:")
            print(X_train.columns.tolist())
            
            # Update training features with previous predictions
            for prev_target in prediction_chain[1:prediction_chain.index(target)]:
                if prev_target in predictions:
                    X_train[prev_target] = predictions[prev_target]['train']
                    X_val[prev_target] = predictions[prev_target]['val']
                    print(f"Added {prev_target} predictions as feature")
            
            print(f"\nAfter adding predictions for {target}, X_train features:")
            print(X_train.columns.tolist())
        
        # Prepare feature matrices
        X_train_current = X_train[current_features].copy()
        X_val_current = X_val[current_features].copy()
        
        # Find optimal parameters
        best_params = find_optimal_params(
            X_train_current, X_val_current,
            y_train[target], y_val[target],
            target, parent_dir, genotype, use_gpu
        )
        
        if best_params is None:
            print(f"Error: Could not find optimal parameters for {target}")
            continue
        
        # Train model with best parameters
        model = train_xgboost_model(
            X_train_current, X_val_current,
            y_train[target], y_val[target],
            best_params, target, parent_dir, genotype,
            use_gpu
        )
        
        # Store model
        models[target] = model
        
        # Make predictions
        train_pred = model.predict(X_train_current)
        val_pred = model.predict(X_val_current)
        
        # Calculate R² scores
        train_r2 = r2_score(y_train[target], train_pred)
        val_r2 = r2_score(y_val[target], val_pred)
        
        print(f"\nPerformance metrics for {target}:")
        print(f"Training R²: {train_r2:.3f}")
        print(f"Validation R²: {val_r2:.3f}")
        
        # Store predictions
        predictions[target] = {
            'train': train_pred,
            'val': val_pred
        }
    
    return models, predictions

def plot_chain_performance(chain_results_dir, predictions, y_val, prediction_chain):
    """Create detailed visualizations of the prediction chain performance."""
    # Calculate actual R² scores
    actual_r2 = {}
    for target in prediction_chain[1:]:
        actual_r2[target] = r2_score(y_val[target], predictions[target]['val'])
    
    # Create R² plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(prediction_chain[1:]))
    plt.bar(x, [actual_r2[t] for t in prediction_chain[1:]], alpha=0.6)
    plt.xlabel('Target')
    plt.ylabel('R² Score')
    plt.title('R² Scores Across Prediction Chain')
    plt.xticks(x, prediction_chain[1:], rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(chain_results_dir / 'r2_scores.png')
    plt.close()
    
    # Create detailed chain progression plot
    plt.figure(figsize=(15, 8))
    
    # Plot R² scores
    plt.plot(prediction_chain[1:], [actual_r2[t] for t in prediction_chain[1:]], 
             'r-', label='R² Score', marker='s')
    plt.grid(True, alpha=0.3)
    plt.ylabel('R² Score')
    plt.title('Prediction Chain Performance')
    plt.legend()
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(chain_results_dir / 'chain_progression.png')
    plt.close()
    
    return actual_r2

def save_prediction_plots_pdf(chain_results_dir, predictions, y_val, prediction_chain, actual_r2):
    """Save all prediction plots in a single PDF file."""
    pdf_path = chain_results_dir / 'prediction_plots.pdf'
    
    with PdfPages(pdf_path) as pdf:
        # Create a summary page
        plt.figure(figsize=(10, 6))
        plt.axis('off')
        plt.text(0.5, 0.9, 'Prediction Results Summary', ha='center', va='center', fontsize=16)
        plt.text(0.1, 0.8, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', fontsize=10)
        plt.text(0.1, 0.7, 'R² Scores:', fontsize=12)
        y_pos = 0.65
        for target in prediction_chain[1:]:
            plt.text(0.2, y_pos, f'{target}: {actual_r2[target]:.3f}', fontsize=10)
            y_pos -= 0.05
        pdf.savefig()
        plt.close()
        
        # Add individual prediction plots
        for target in prediction_chain[1:]:
            plt.figure(figsize=(10, 6))
            
            # Scatter plot
            plt.scatter(y_val[target], predictions[target]['val'], alpha=0.5)
            
            # Perfect prediction line
            min_val = min(y_val[target].min(), predictions[target]['val'].min())
            max_val = max(y_val[target].max(), predictions[target]['val'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            # Labels and title
            plt.xlabel(f'Actual {target}')
            plt.ylabel(f'Predicted {target}')
            plt.title(f'{target} Predictions\nR² Score: {actual_r2[target]:.3f}')
            
            # Add grid
            plt.grid(True, alpha=0.3)
            
            # Add correlation info
            plt.text(0.05, 0.95, f'R² = {actual_r2[target]:.3f}',
                    transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
        
        # Add R² comparison plot
        plt.figure(figsize=(12, 6))
        x = np.arange(len(prediction_chain[1:]))
        plt.bar(x, [actual_r2[t] for t in prediction_chain[1:]], alpha=0.6)
        plt.xlabel('Target')
        plt.ylabel('R² Score')
        plt.title('R² Scores Across Prediction Chain')
        plt.xticks(x, prediction_chain[1:], rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

def plot_trial_predictions(predictions, y_val, trial_indices, target, output_dir):
    """Create line plots comparing predicted vs actual values for each trial."""
    print(f"\nCreating trial plots for {target}...")
    
    # Create a directory for trial plots
    trial_plots_dir = output_dir / 'trial_plots'
    trial_plots_dir.mkdir(exist_ok=True)
    
    # Get unique trials
    unique_trials = np.unique(trial_indices)
    print(f"Found {len(unique_trials)} unique trials")
    
    # Verify data exists
    if target not in predictions or target not in y_val:
        print(f"Error: Missing data for target {target}")
        print("Available targets in predictions:", list(predictions.keys()))
        print("Available targets in y_val:", list(y_val.columns))
        return
    
    # Create PDF for all trials
    pdf_path = trial_plots_dir / f'{target}_trial_predictions.pdf'
    print(f"Creating PDF: {pdf_path}")
    
    with PdfPages(pdf_path) as pdf:
        # Summary page
        plt.figure(figsize=(10, 6))
        plt.axis('off')
        plt.text(0.5, 0.9, f'Trial Predictions for {target}', ha='center', va='center', fontsize=16)
        plt.text(0.1, 0.8, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', fontsize=10)
        plt.text(0.1, 0.7, f'Number of trials: {len(unique_trials)}', fontsize=12)
        plt.text(0.1, 0.6, f'Frames per trial: 600 (frames 400-1000)', fontsize=12)
        pdf.savefig()
        plt.close()
        
        # Plot each trial
        for trial_idx, trial in enumerate(unique_trials):
            print(f"Processing trial {trial_idx + 1}/{len(unique_trials)} for {target}")
            
            trial_mask = trial_indices == trial
            
            # Get trial data
            actual = y_val[target][trial_mask].values
            predicted = predictions[target]['val'][trial_mask]
            
            if len(actual) == 0 or len(predicted) == 0:
                print(f"Warning: No data for trial {trial} in {target}")
                continue
            
            if len(actual) != 600:  # Should be 600 frames (400-1000)
                print(f"Warning: Trial {trial} has {len(actual)} frames instead of 600")
                continue
            
            # Create plot
            plt.figure(figsize=(15, 8))
            
            # Plot actual and predicted values
            frames = np.arange(400, 1000)  # Actual frame numbers
            plt.plot(frames, actual, 'b-', label='Actual', linewidth=2, alpha=0.7)
            plt.plot(frames, predicted, 'r--', label='Predicted', linewidth=2, alpha=0.7)
            
            # Add confidence band
            residuals = np.abs(actual - predicted)
            conf_band = np.std(residuals) * 1.96  # 95% confidence interval
            plt.fill_between(frames, 
                           predicted - conf_band,
                           predicted + conf_band,
                           color='red', alpha=0.1, label='95% Confidence')
            
            # Calculate metrics for this trial
            trial_r2 = r2_score(actual, predicted)
            trial_mae = mean_absolute_error(actual, predicted)
            trial_rmse = np.sqrt(mean_squared_error(actual, predicted))
            
            # Add labels and title
            plt.xlabel('Frame Number', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.title(f'{target} - Trial {trial + 1}\nR² = {trial_r2:.3f}, MAE = {trial_mae:.3f}, RMSE = {trial_rmse:.3f}',
                     fontsize=14, pad=20)
            
            # Add grid
            plt.grid(True, alpha=0.3, linestyle='--')
            
            # Add legend
            plt.legend(fontsize=10, loc='upper right')
            
            # Add metrics text box
            metrics_text = (f'R² = {trial_r2:.3f}\n'
                          f'MAE = {trial_mae:.3f}\n'
                          f'RMSE = {trial_rmse:.3f}')
            plt.text(0.02, 0.98, metrics_text,
                    transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                    verticalalignment='top',
                    fontsize=10)
            
            # Customize axis
            plt.xticks(np.arange(400, 1001, 100))
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
        
        # Add a summary plot with all trials overlaid
        plt.figure(figsize=(20, 10))
        for trial in unique_trials:
            trial_mask = trial_indices == trial
            actual = y_val[target][trial_mask].values
            predicted = predictions[target]['val'][trial_mask]
            frames = np.arange(400, 1000)
            plt.plot(frames, actual, 'b-', alpha=0.2)
            plt.plot(frames, predicted, 'r--', alpha=0.2)
        
        plt.xlabel('Frame Number', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title(f'{target} - All Trials Overlay\nBlue: Actual, Red: Predicted', fontsize=14, pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(np.arange(400, 1001, 100))
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
    print(f"Completed trial plots for {target}")

def main():
    """Main function to run XGBoost analysis."""
    print("\nStarting XGBoost analysis...")
    
    # Check GPU availability
    use_gpu = check_gpu_availability()
    
    # Create output directories
    parent_dir = Path('xgboost_results')
    parent_dir.mkdir(exist_ok=True)
    
    # Configuration
    window_size = 100  # Number of frames of context needed before prediction
    
    # List of genotypes to process
    genotypes = ['P9LT']  # Focus on P9LT for prediction chain
    
    # Define base features used across all predictions
    base_features = [
        'x_vel_ma5',    # Moving average velocities
        'y_vel_ma5',
        'z_vel_ma5',
        'x_vel_ma10',
        'y_vel_ma10',
        'z_vel_ma10',
        'x_vel_ma20',
        'y_vel_ma20',
        'z_vel_ma20',
        'x_vel',        # Raw velocities
        'y_vel',
        'z_vel'
    ]
    
    # Define prediction chain configuration
    prediction_chain = [
        'R-F-CTr_y',    # Starting point - ground truth
        'R1A_flex',     # First prediction
        'R-F-FeTi_y',   # Then predict TaG positions
        'R-F-TiTa_y',
        'R-F-TaG_y',
        'R1B_flex',     # Then rest of joint angles
        'R1A_abduct',
        'R1C_flex',
        'R1A_rot',
        'R1D_flex',
        'R1B_rot',
        'R1C_rot'
    ]
    
    # Features specific to each target in the chain
    chain_features = {
        # R1A_flex - strongest correlations with CTr_y (0.971)
        'R1A_flex': ['R-F-CTr_y'] + base_features,
        
        # R-F-FeTi_y - uses R-F-CTr_y and R1A_flex
        'R-F-FeTi_y': ['R-F-CTr_y', 'R1A_flex'] + base_features,
        
        # R-F-TiTa_y - uses CTr_y, FeTi_y
        'R-F-TiTa_y': ['R-F-CTr_y', 'R-F-FeTi_y'] + base_features,
        
        # R-F-TaG_y - uses CTr_y, FeTi_y, TiTa_y
        'R-F-TaG_y': ['R-F-CTr_y', 'R-F-FeTi_y', 'R-F-TiTa_y'] + base_features,
        
        # R1B_flex - strong correlation with FeTi_z (-0.933)
        'R1B_flex': ['R-F-FeTi_y', 'R1A_flex'] + base_features,
        
        # R1A_abduct - strong correlation with CTr_y (-0.819)
        'R1A_abduct': ['R-F-CTr_y', 'R1A_flex'] + base_features,
        
        # R1C_flex - correlates with FeTi_z (-0.801)
        'R1C_flex': ['R-F-FeTi_y', 'R1B_flex'] + base_features,
        
        # R1A_rot - correlates with CTr_y (0.764) and FeTi_y (0.897)
        'R1A_rot': ['R-F-CTr_y', 'R-F-FeTi_y', 'R1A_flex'] + base_features,
        
        # R1D_flex - chain dependency from previous flex angles
        'R1D_flex': ['R1A_flex', 'R1B_flex', 'R1C_flex'] + base_features,
        
        # R1B_rot - chain dependency from flex angles
        'R1B_rot': ['R1B_flex', 'R-F-FeTi_y'] + base_features,
        
        # R1C_rot - chain dependency from flex angles
        'R1C_rot': ['R1C_flex', 'R-F-FeTi_y'] + base_features
    }
    
    for genotype in genotypes:
        print(f"\nProcessing {genotype}...")
        genotype_dir = parent_dir / genotype
        genotype_dir.mkdir(exist_ok=True)
        
        # Prepare data
        df, features, targets, (train_mask, val_mask, test_mask) = prepare_data_for_xgboost(genotype, window_size=window_size)
        if df is None:
            print(f"Skipping {genotype} due to data preparation error")
            continue
        
        X_train, X_val, X_test, y_train, y_val, y_test = df, features, targets, train_mask, val_mask, test_mask
        
        if genotype == 'P9LT':
            # Train prediction chain
            print("\nTraining prediction chain for P9LT...")
            print("Starting with R-F-CTr_y to predict joint angles...")
            
            # Train the chain
            models, predictions = train_prediction_chain(
                X_train.copy(), X_val.copy(), y_train.copy(), y_val.copy(),
                features, prediction_chain, chain_features,
                genotype_dir, genotype, use_gpu
            )
            
            # Save chain results
            chain_results_dir = genotype_dir / 'chain_results'
            chain_results_dir.mkdir(exist_ok=True)
            
            # Create performance visualizations
            actual_r2 = plot_chain_performance(
                chain_results_dir, predictions, y_val, prediction_chain
            )
            
            # Save prediction plots in PDF
            save_prediction_plots_pdf(
                chain_results_dir, predictions, y_val,
                prediction_chain, actual_r2
            )
            
            # Save predictions
            for target in prediction_chain[1:]:  # Skip R-F-CTr_y (input)
                target_dir = chain_results_dir / target
                target_dir.mkdir(exist_ok=True)
                
                # Save model
                model = models[target]
                model.save_model(str(target_dir / f'{target}_model.json'))
                
                # Create trial-by-trial prediction plots
                plot_trial_predictions(
                    predictions, y_val, 
                    df[val_mask]['trial_id'].values,  # Use actual trial IDs
                    target, target_dir
                )
                
                # Plot actual vs predicted (individual PNG files)
                plt.figure(figsize=(10, 6))
                plt.scatter(y_val[target], predictions[target]['val'], alpha=0.5)
                plt.plot([y_val[target].min(), y_val[target].max()],
                        [y_val[target].min(), y_val[target].max()],
                        'r--', lw=2)
                plt.xlabel(f'Actual {target}')
                plt.ylabel(f'Predicted {target}')
                plt.title(f'{target} Predictions\nR² Score: {actual_r2[target]:.3f}')
                plt.savefig(target_dir / 'validation_scatter.png')
                plt.close()
            
            print("\nPrediction chain training completed!")
            print(f"Results saved to: {chain_results_dir}")
            
        else:
            # Original training process for other genotypes
            for target in targets:
                best_params = find_optimal_params(
                    X_train, X_val, y_train[target], y_val[target],
                    target, genotype_dir, genotype, use_gpu
                )
                
                if best_params is None:
                    print(f"Skipping {target} due to parameter optimization error")
                    continue
                
                train_xgboost_model(
                    X_train, X_val, y_train[target], y_val[target],
                    best_params, target, genotype_dir, genotype, use_gpu
                )
    
    print("\nXGBoost analysis completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train XGBoost models for joint angle prediction')
    parser.add_argument('--genotype', type=str, default='BPN', help='Genotype to train on')
    parser.add_argument('--window_size', type=int, default=100, help='Window size for context')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for training')
    
    args = parser.parse_args()
    main(args.genotype, args.window_size, args.use_gpu) 