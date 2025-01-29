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

def prepare_data_for_xgboost(genotype):
    """Prepare data for XGBoost model using top correlated features."""
    # Load the preprocessed data
    data_path = "/Users/anivenkat/Downloads/BPN_P9LT_P9RT_flyCoords.csv"
    try:
        df = load_and_filter_data(data_path, genotype)
    except FileNotFoundError:
        print(f"Could not find data file at {data_path}")
        print("Please update the path to your data file")
        return None, None
    
    # Select features and targets based on correlation analysis for each genotype
    if genotype == 'P9RT':
        features = ['x_vel', 'y_vel', 'z_vel']  # Simplified feature set
        targets = ['L2B_flex', 'L1B_rot']
    elif genotype == 'BPN':
        features = ['x_vel', 'y_vel', 'z_vel']  # Simplified feature set
        targets = ['R3A_flex', 'L2C_flex']
    else:  # P9LT
        features = ['x_vel', 'y_vel', 'z_vel']  # Simplified feature set
        targets = ['L2B_rot', 'L1C_rot']
    
    print(f"\nUsing features for {genotype}:")
    for feat in features:
        print(f"  {feat}")
    print(f"\nPredicting targets:")
    for target in targets:
        print(f"  {target}")
    
    # Prepare feature matrix X and target matrix y
    X = df[features]
    y = df[targets]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return (X_train, X_test, y_train, y_test), (features, targets)

def find_optimal_params(X_train, y_train, target_name, parent_dir, genotype):
    """Find optimal hyperparameters using cross-validation."""
    print(f"\nFinding optimal parameters for {target_name}...")
    
    # Parameter combinations to try with more comprehensive search space
    param_combinations = [
        {
            'max_depth': max_depth,
            'learning_rate': lr,
            'n_estimators': n_est,
            'min_child_weight': mcw,
            'subsample': ss,
            'colsample_bytree': cs,
            'gamma': g,
            'reg_alpha': alpha,
            'reg_lambda': lamb
        }
        for max_depth in [7, 9, 11]  # More options for tree depth
        for lr in [0.01]  # More learning rate options
        for n_est in [2000]  # Different numbers of estimators
        for mcw in [1, 3, 5]  # More min_child_weight options
        for ss in [0.6]  # More subsample ratios
        for cs in [0.6]  # More colsample_bytree ratios
        for g in [0, 0.1, 0.2]  # More gamma options
        for alpha in [0, 0.1, 1.0]  # L1 regularization
        for lamb in [0, 0.1, 1.0]  # L2 regularization
    ]
    
    # Calculate total number of combinations
    total_combinations = len(param_combinations)
    print(f"Total parameter combinations to try: {total_combinations}")
    
    # Create directory for parameter search results
    param_search_dir = parent_dir / genotype / 'parameter_search' / target_name
    param_search_dir.mkdir(parents=True, exist_ok=True)
    
    best_mae = float('inf')
    best_params = None
    best_model = None
    param_search_results = []  # Renamed from results to param_search_results
    
    # Try each parameter combination with progress bar
    for i, params in enumerate(tqdm(param_combinations, desc="Parameter combinations", unit="combination")):
        # Use 5-fold cross-validation with progress bar
        cv_scores = []
        cv_models = []
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror',  # Changed from absoluteerror to get better convergence
            early_stopping_rounds=20,  # Reduced to prevent overfitting
            eval_metric=['mae', 'rmse'],  # Track both metrics
            **params
        )
        
        # Create progress bar for cross-validation folds
        for fold in tqdm(range(5), desc=f"CV Folds for combination {i+1}/{total_combinations}", leave=False):
            # Split data for this fold
            split_idx = len(X_train) // 5
            val_start = fold * split_idx
            val_end = (fold + 1) * split_idx if fold < 4 else len(X_train)
            
            X_val = X_train.iloc[val_start:val_end]
            y_val = y_train.iloc[val_start:val_end]
            X_train_fold = pd.concat([X_train.iloc[:val_start], X_train.iloc[val_end:]])
            y_train_fold = pd.concat([y_train.iloc[:val_start], y_train.iloc[val_end:]])
            
            # Create evaluation set for early stopping
            eval_set = [(X_train_fold, y_train_fold), (X_val, y_val)]
            
            # Train and evaluate
            model.fit(X_train_fold, y_train_fold,
                     eval_set=eval_set,
                     verbose=False)
            
            # Get validation score using MAE
            val_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, val_pred)
            cv_scores.append(mae)
            cv_models.append(model)
        
        # Calculate average MAE across folds
        avg_mae = np.mean(cv_scores)
        std_mae = np.std(cv_scores)
        
        param_search_results.append({
            'params': params,
            'mae': avg_mae,
            'std': std_mae
        })
        
        # Update best parameters if we found better ones
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_params = params
            best_model = cv_models[np.argmin(cv_scores)]  # Keep the best model from CV
            
            # Save learning curve only for the best model so far
            plt.figure(figsize=(10, 6))
            # Plot both MAE and RMSE curves
            eval_results = best_model.evals_result()  # Renamed from results to eval_results
            plt.plot(eval_results['validation_1']['mae'], label='Validation MAE', color='blue')
            plt.plot(eval_results['validation_1']['rmse'], label='Validation RMSE', color='red', alpha=0.7)
            plt.title(f'Learning Curves - Best Model\nMAE: {best_mae:.4f}')
            plt.xlabel('Boosting Round')
            plt.ylabel('Error')
            plt.legend()
            plt.grid(True)
            plt.savefig(param_search_dir / 'best_model_learning_curve.png')
            plt.close()
    
    # Save all results to CSV
    results_df = pd.DataFrame([
        {**r['params'], 'mae': r['mae'], 'std': r['std']}
        for r in param_search_results  # Updated to use param_search_results
    ])
    results_df.to_csv(param_search_dir / 'parameter_search_results.csv', index=False)
    
    # Plot parameter importance
    param_importance = {}
    for param in best_params.keys():
        values = results_df[param].unique()
        scores = [results_df[results_df[param] == v]['mae'].mean() for v in values]
        param_importance[param] = np.std(scores)  # Higher std means parameter is more important
    
    plt.figure(figsize=(10, 6))
    plt.bar(param_importance.keys(), param_importance.values())
    plt.title('Parameter Importance\n(Higher value = More important)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(param_search_dir / 'parameter_importance.png')
    plt.close()
    
    print(f"\nBest parameters for {target_name}:")
    print(best_params)
    print(f"Best MAE: {best_mae:.4f}")
    
    return best_params

def train_xgboost_models(X_train, y_train, features, targets, genotype, parent_dir):
    """Train separate XGBoost models for each target variable with optimized parameters."""
    models = {}
    
    for target in targets:
        print(f"\nTraining model for {target}...")
        
        # Find optimal parameters
        best_params = find_optimal_params(X_train, y_train[target], target, parent_dir, genotype)
        
        # Train final model with best parameters and even more epochs
        final_params = best_params.copy()
        final_params['n_estimators'] = 10000  # Significantly more epochs for final training
        
        model = xgb.XGBRegressor(
            objective='reg:absoluteerror',
            early_stopping_rounds=50,  # Increased early stopping rounds for longer training
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
        
        # Train models
        print(f"\nTraining XGBoost models for {genotype}...")
        models = train_xgboost_models(X_train, y_train, features, targets, genotype, parent_dir)
        
        # Evaluate models
        print(f"\nEvaluating models for {genotype}...")
        results = evaluate_models(models, X_test, y_test, features, targets, genotype, parent_dir)
        
        # Save results
        save_results(results, features, targets, genotype, parent_dir)
        print(f"\nResults saved to {parent_dir}/{genotype}/")

if __name__ == "__main__":
    main() 