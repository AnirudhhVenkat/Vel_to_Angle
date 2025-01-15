import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(outputs, targets):
    """Calculate multiple metrics for time series prediction"""
    # Convert to numpy if tensors
    outputs = outputs.cpu().detach().numpy() if torch.is_tensor(outputs) else outputs
    targets = targets.cpu().detach().numpy() if torch.is_tensor(targets) else targets
    
    metrics = {}
    
    # MSE and RMSE
    metrics['mse'] = mean_squared_error(targets, outputs)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # MAE
    metrics['mae'] = mean_absolute_error(targets, outputs)
    
    # MAPE (Mean Absolute Percentage Error)
    epsilon = 1e-10  # Small constant to avoid division by zero
    mape = np.mean(np.abs((targets - outputs) / (targets + epsilon))) * 100
    metrics['mape'] = mape
    
    # R-squared (Coefficient of Determination)
    ss_res = np.sum((targets - outputs) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    metrics['r2'] = 1 - (ss_res / (ss_tot + epsilon))
    
    # Dynamic Time Warping Distance (simplified version)
    def dtw_distance(y_true, y_pred):
        n, m = len(y_true), len(y_pred)
        dtw_matrix = np.full((n+1, m+1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = np.abs(y_true[i-1] - y_pred[j-1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # insertion
                                            dtw_matrix[i, j-1],    # deletion
                                            dtw_matrix[i-1, j-1])  # match
        return dtw_matrix[n, m]
    
    # Calculate DTW for each dimension and average
    dtw_scores = []
    for i in range(outputs.shape[1]):
        dtw = dtw_distance(targets[:, i], outputs[:, i])
        dtw_scores.append(dtw)
    metrics['dtw'] = np.mean(dtw_scores)
    
    return metrics