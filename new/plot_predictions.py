import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

from models.unsupervised_transformer import UnsupervisedTransformerModel
from utils.data import prepare_data

def plot_predictions(predictions, targets, section_size=600):
    """Plot predictions vs targets for each joint angle."""
    joint_names = [
        'L2B_rot', 'R3A_rot', 'R3A_flex', 
        'R1B_rot', 'R2B_rot', 'L3A_rot'
    ]
    
    # Convert inputs to numpy if they're tensors
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    
    # Calculate number of complete sections
    total_timesteps = len(predictions)
    num_sections = total_timesteps // section_size
    if total_timesteps % section_size > 0:
        num_sections += 1
    
    print(f"Total timesteps: {total_timesteps}")
    print(f"Number of sections: {num_sections}")
    
    # Create a separate figure for each joint angle
    for joint_idx, joint_name in enumerate(joint_names):
        fig_height = min(3 * num_sections, 25)
        fig = plt.figure(figsize=(12, fig_height))
        fig.suptitle(f'{joint_name}', y=1.02)
        
        # Plot each section as a subplot
        for section in range(num_sections):
            start_idx = section * section_size
            end_idx = min((section + 1) * section_size, total_timesteps)
            
            if end_idx - start_idx < 10:
                continue
            
            ax = plt.subplot(num_sections, 1, section + 1)
            
            section_predictions = predictions[start_idx:end_idx, joint_idx]
            section_targets = targets[start_idx:end_idx, joint_idx]
            
            stride = max(1, len(section_predictions) // 100)
            x_values = np.arange(0, len(section_predictions), stride)
            
            ax.plot(x_values, section_targets[::stride], 'b-', label='Target', alpha=0.7, linewidth=1.5)
            ax.plot(x_values, section_predictions[::stride], 'r-', label='Prediction', alpha=0.7, linewidth=1.5)
            
            # Calculate metrics
            mse = np.mean((section_predictions - section_targets) ** 2)
            mae = np.mean(np.abs(section_predictions - section_targets))
            r2 = stats.pearsonr(section_predictions, section_targets)[0] ** 2
            
            ax.set_title(f'Trial {section + 1} (MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f})')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Angle (degrees)')
            ax.grid(True, alpha=0.3)
            
            if section == 0:
                ax.legend()
            
            y_min = min(section_targets.min(), section_predictions.min())
            y_max = max(section_targets.max(), section_predictions.max())
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            
            num_ticks = 5
            tick_positions = np.linspace(0, len(section_predictions), num_ticks, dtype=int)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f'{x+start_idx}' for x in tick_positions])
        
        plt.tight_layout()
        plt.show()

def main():
    # Set the path to your saved model
    model_path = "path/to/your/model.pth"  # Change this to your model path
    
    # Load the saved model
    checkpoint = torch.load(model_path)
    config = checkpoint['config']
    model_state = checkpoint['model_state_dict']
    
    print("Model Configuration:")
    for key, value in config['model_params'].items():
        print(f"{key}: {value}")
    
    print(f"\nBest Loss: {checkpoint['best_loss']}")
    print(f"Best Epoch: {checkpoint['best_epoch']}")
    
    # Initialize model with the same configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device
    
    model = UnsupervisedTransformerModel(
        input_size=config['model_params']['input_size'],
        hidden_size=max(config['model_params']['hidden_sizes']),
        nhead=config['model_params']['nhead'],
        num_layers=config['model_params']['num_layers'],
        output_size=6,  # For joint angle prediction
        dropout=config['model_params']['dropout']
    ).to(device)
    
    # Load the state dict
    model.load_state_dict(model_state)
    model.eval()
    print("Model loaded successfully!")
    
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(config)
    print("Data loaded successfully!")
    
    # Generate predictions
    test_predictions = []
    test_targets = []
    window_size = config['window_size']
    
    with torch.no_grad():
        for i in range(0, len(test_loader.dataset), window_size):
            end_idx = min(i + window_size, len(test_loader.dataset))
            window_inputs = test_loader.dataset[i:end_idx][0]
            window_targets = test_loader.dataset[i:end_idx][1]
            
            if len(window_inputs) == window_size:
                window_inputs = window_inputs.to(device)
                outputs = model(window_inputs)
                test_predictions.append(outputs.cpu())
                test_targets.append(window_targets)
    
    test_predictions = torch.cat(test_predictions, dim=0).numpy()
    test_targets = torch.cat(test_targets, dim=0).numpy()
    
    print(f"Prediction shape: {test_predictions.shape}")
    print(f"Target shape: {test_targets.shape}")
    
    # Plot predictions
    plot_predictions(test_predictions, test_targets)

if __name__ == "__main__":
    main() 