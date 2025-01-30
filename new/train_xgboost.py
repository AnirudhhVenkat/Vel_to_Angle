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

def prepare_data_for_xgboost(genotype):
    """Prepare data for XGBoost model using top correlated features."""
    # Load the preprocessed data
    try:
        data_path = get_available_data_path()
        df = load_and_filter_data(data_path, genotype)
    except FileNotFoundError:
        print("Could not find the data file in any of the expected locations")
        print("Please ensure the data file is available in one of the expected locations")
        return None, None
    
    # Print available columns for debugging
    print("\nAvailable columns in DataFrame:")
    for col in sorted(df.columns):
        print(f"  {col}")
    
    # Select features and targets based on correlation analysis for each genotype
    if genotype == 'P9RT':
        features = [
            'L-F-TaG_y',     # -0.815 with L1B_flex
            'x_vel',         # Velocity components
            'y_vel',
            'z_vel'
        ]
        
        targets = [
            'L1A_flex'      # Primary target
        ]
        
    elif genotype == 'BPN':
        features = [
            'L-F-TaG_y',     # Strong correlation with L1A_flex
            'x_vel',         # Velocity components
            'y_vel',
            'z_vel'
        ]
        
        targets = [
            'L1A_flex'      # Primary target
        ]
        
    else:  # P9LT
        features = [
            'R-F-TaG_y',     # Strong correlation with R1A_flex
            'x_vel',         # Velocity components
            'y_vel',
            'z_vel'
        ]
        
        targets = [
            'R1A_flex'      # Primary target
        ]
    
    # Verify all features exist in DataFrame
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"\nError: The following features are missing from the DataFrame:")
        for feat in missing_features:
            print(f"  {feat}")
        return None, None
    
    # Verify all targets exist in DataFrame
    missing_targets = [t for t in targets if t not in df.columns]
    if missing_targets:
        print(f"\nError: The following targets are missing from the DataFrame:")
        for targ in missing_targets:
            print(f"  {targ}")
        return None, None
    
    print(f"\nUsing features for {genotype}:")
    for feat in features:
        print(f"  {feat}")
    print(f"\nPredicting targets:")
    for targ in targets:
        print(f"  {targ}")
    
    # Prepare feature matrix X and target matrix y
    X = df[features]
    y = df[targets]
    
    # Print shapes for debugging
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target matrix shape: {y.shape}")
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return (X_train, X_test, y_train, y_test), (features, targets)

def find_optimal_params(X_train, y_train, target_name, parent_dir, genotype, use_gpu=False):
    """Find optimal hyperparameters using cross-validation."""
    print(f"\nFinding optimal parameters for {target_name}...")
    
    # Base parameters for grid search
    base_params = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [1000],
        'min_child_weight': [1, 3, 5],
        'objective': ['reg:absoluteerror'],
        'tree_method': ['hist']
    }
    
    # Add GPU device if available
    if use_gpu:
        base_params['device'] = ['cuda']
    
    # Create parameter grid
    param_grid = base_params.copy()
    
    # Create output directory for this genotype
    output_dir = parent_dir / genotype
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Lists to store results
    param_search_results = []
    
    # Perform grid search with cross-validation
    print("\nPerforming grid search with cross-validation...")
    
    # Create parameter combinations
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                        for v in itertools.product(*param_grid.values())]
    
    # Progress bar for parameter combinations
    for params in tqdm(param_combinations, desc="Parameter combinations"):
        # Create and configure model
        model = xgb.XGBRegressor(**params)
        
        # Perform k-fold cross-validation
        cv_scores = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Progress bar for cross-validation folds
        for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X_train), 
                                                       total=5, 
                                                       desc="Cross-validation",
                                                       leave=False)):
            X_train_fold = X_train.iloc[train_idx]
            y_train_fold = y_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_val_fold = y_train.iloc[val_idx]
            
            # Create evaluation set for early stopping
            eval_set = [(X_val_fold, y_val_fold)]
            
            # Train model
            model.fit(X_train_fold, y_train_fold,
                     eval_set=eval_set,
                     eval_metric=['mae', 'rmse'],
                     early_stopping_rounds=20,
                     verbose=False)
            
            # Make predictions
            y_pred = model.predict(X_val_fold)
            
            # Calculate MAE for this fold
            mae = mean_absolute_error(y_val_fold, y_pred)
            cv_scores.append(mae)
        
        # Calculate mean and std of cross-validation scores
        mean_mae = np.mean(cv_scores)
        std_mae = np.std(cv_scores)
        
        # Store results
        param_search_results.append({
            'params': params,
            'mean_mae': mean_mae,
            'std_mae': std_mae
        })
    
    # Convert results to DataFrame and sort by mean MAE
    results_df = pd.DataFrame(param_search_results)
    results_df = results_df.sort_values('mean_mae')
    
    # Save results
    results_df.to_csv(output_dir / f'parameter_search_results_{target_name}.csv', index=False)
    
    # Get best parameters
    best_params = results_df.iloc[0]['params']
    best_mae = results_df.iloc[0]['mean_mae']
    
    print(f"\nBest parameters for {target_name}:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best Mean MAE: {best_mae:.4f}")
    
    return best_params

def train_xgboost_models(X_train, y_train, features, targets, genotype, parent_dir, use_gpu=False):
    """Train separate XGBoost models for each target variable with optimized parameters."""
    models = {}
    
    for target in targets:
        print(f"\nTraining model for {target}...")
        
        # Find optimal parameters
        best_params = find_optimal_params(X_train, y_train[target], target, parent_dir, genotype, use_gpu)
        
        # Train final model with best parameters and even more epochs
        final_params = best_params.copy()
        final_params['n_estimators'] = 10000  # Significantly more epochs for final training
        final_params['tree_method'] = 'hist'  # Always use hist method
        final_params['objective'] = 'reg:absoluteerror'  # Ensure MAE objective
        
        # Add GPU device if available
        if use_gpu:
            final_params['device'] = 'cuda'
        
        model = xgb.XGBRegressor(
            early_stopping_rounds=50,
            eval_metric='mae',
            **final_params
        )
        
        # Train with a validation set for early stopping
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train[target], test_size=0.2, random_state=42
        )
        
        print(f"Training final model for {target} with {final_params['n_estimators']} max epochs...")
        model.fit(
            X_train_final, y_train_final,
            eval_set=[(X_val, y_val)],
            verbose=True  # Show progress for final training
        )
        
        # Print actual number of trees used (after early stopping)
        print(f"Final number of trees used: {model.get_booster().num_boosted_rounds()}")
        
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
    plt.style.use('bmh')
    
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
        trial_size = 600
        num_trials = min(4, len(predictions) // trial_size)  # Show up to 4 trials
        
        # Create subplot figure
        fig, axes = plt.subplots(num_trials, 1, figsize=(40, 6*num_trials), dpi=300)  # Increased width from 30 to 40
        fig.suptitle(f'Predicted vs Actual {target} ({genotype})\nR² = {r2:.3f}, RMSE = {rmse:.3f}', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Plot each trial in a separate subplot
        for trial in range(num_trials):
            start_idx = trial * trial_size
            end_idx = start_idx + trial_size
            
            ax = axes[trial] if num_trials > 1 else axes
            
            # Plot actual and predicted values for this trial with thicker lines
            ax.plot(range(trial_size), 
                   y_test[target].iloc[start_idx:end_idx],
                   label='Actual', color='#2E86C1', linewidth=2.0, alpha=0.8)  # Increased linewidth from 1.5 to 2.0
            ax.plot(range(trial_size), 
                   predictions[start_idx:end_idx],
                   label='Predicted', color='#E74C3C', linewidth=2.0, alpha=0.8)  # Increased linewidth from 1.5 to 2.0
            
            # Enhanced styling for each subplot
            ax.set_xlabel('Time Points', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{target} Value', fontsize=12, fontweight='bold')
            ax.set_title(f'Trial {trial + 1}', fontsize=12, fontweight='bold', pad=10)
            ax.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.minorticks_on()
            ax.grid(True, which='minor', linestyle=':', alpha=0.4)
            
            # Add trial-specific metrics
            trial_mse = mean_squared_error(y_test[target].iloc[start_idx:end_idx], 
                                         predictions[start_idx:end_idx])
            trial_r2 = r2_score(y_test[target].iloc[start_idx:end_idx], 
                              predictions[start_idx:end_idx])
            ax.text(0.02, 0.95, f'Trial R² = {trial_r2:.3f}\nTrial RMSE = {np.sqrt(trial_mse):.3f}',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(facecolor='white', alpha=0.8))
        
        # Improve overall layout
        plt.tight_layout()
        
        # Save high-quality plot
        output_dir = parent_dir / genotype
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / f'predictions_vs_actual_trials_{target}.png',
                   dpi=300, bbox_inches='tight', facecolor='white')
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

def main():
    """Main function to run XGBoost analysis."""
    # Check GPU availability first
    use_gpu = check_gpu_availability()
    
    genotypes = ['P9RT', 'BPN', 'P9LT']
    
    # Create parent directory for all results
    parent_dir = Path("xgboost_results")
    parent_dir.mkdir(parents=True, exist_ok=True)
    
    for genotype in genotypes:
        print(f"\nProcessing {genotype} genotype...")
        
        # Prepare data
        data, (features, targets) = prepare_data_for_xgboost(genotype)
        if data is None:
            continue
        
        X_train, X_test, y_train, y_test = data
        
        # Train models with GPU if available
        print(f"\nTraining XGBoost models for {genotype} using {'GPU' if use_gpu else 'CPU'}...")
        models = train_xgboost_models(X_train, y_train, features, targets, genotype, parent_dir, use_gpu)
        
        # Evaluate models
        print(f"\nEvaluating models for {genotype}...")
        results = evaluate_models(models, X_test, y_test, features, targets, genotype, parent_dir)
        
        # Save results
        save_results(results, features, targets, genotype, parent_dir)
        print(f"\nResults saved to {parent_dir}/{genotype}/")

if __name__ == "__main__":
    main() 