import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import json
from losses import create_derivative_loss, create_weighted_mae_loss, create_combined_loss
from datetime import datetime

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class HybridLSTMTransformer(nn.Module):
    """Hybrid model combining LSTM and Transformer architectures"""
    def __init__(self, input_size, hidden_size, output_size, 
                 num_lstm_layers=2, num_transformer_layers=2, 
                 nhead=8, dropout=0.1, lstm_ratio=0.7):
        """
        Args:
            input_size (int): Number of input features
            hidden_size (int): Hidden dimension size
            output_size (int): Number of output features
            num_lstm_layers (int): Number of LSTM layers
            num_transformer_layers (int): Number of Transformer layers
            nhead (int): Number of Transformer attention heads
            dropout (float): Dropout rate
            lstm_ratio (float): Ratio of LSTM to Transformer hidden size (0-1)
        """
        super(HybridLSTMTransformer, self).__init__()
        
        # Calculate split hidden sizes based on ratio
        self.lstm_hidden = int(hidden_size * lstm_ratio)
        self.transformer_hidden = hidden_size - self.lstm_hidden
        
        # LSTM component
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.lstm_hidden,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Transformer component
        self.pos_encoder = PositionalEncoding(self.transformer_hidden)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.transformer_hidden,
            nhead=nhead,
            dim_feedforward=self.transformer_hidden * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_transformer_layers
        )
        
        # Input projection for transformer
        self.input_projection = nn.Linear(input_size, self.transformer_hidden)
        
        # Combine LSTM and Transformer outputs
        self.output_layer = nn.Sequential(
            nn.Linear(self.lstm_hidden + self.transformer_hidden, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        """
        Forward pass through the hybrid model
        
        Args:
            x (torch.Tensor): Input tensor [batch, seq_len, features]
            
        Returns:
            torch.Tensor: Output predictions [batch, seq_len, output_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Transformer processing
        transformer_input = self.input_projection(x)
        transformer_input = self.pos_encoder(transformer_input)
        transformer_out = self.transformer_encoder(transformer_input)
        
        # Combine LSTM and Transformer outputs
        combined = torch.cat([lstm_out, transformer_out], dim=-1)
        
        # Generate predictions
        output = self.output_layer(combined)
        
        return output

class SequenceDataset(Dataset):
    """Dataset for sequence prediction"""
    def __init__(self, X, y, sequence_length=50):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.X) - self.sequence_length
    
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.sequence_length]
        y_seq = self.y[idx:idx + self.sequence_length]
        return X_seq, y_seq

def prepare_data(data_path, input_features, output_features, sequence_length, genotype='ALL'):
    """Load and prepare data for training"""
    print("\nPreparing data...")
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded data with shape: {df.shape}")
    
    # Filter for specified genotype if not ALL
    if genotype != 'ALL':
        df = df[df['genotype'] == genotype].copy()
        print(f"Filtered for {genotype}: {df.shape}")
    
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
    
    # Calculate moving averages for velocities within each trial
    base_velocities = ['x_vel', 'y_vel', 'z_vel']
    for window in [5, 10, 20]:
        for vel in base_velocities:
            # Calculate moving average for each trial separately
            ma_values = []
            for trial in range(num_trials):
                start_idx = trial * trial_size
                end_idx = (trial + 1) * trial_size
                trial_data = complete_trials_data[vel].iloc[start_idx:end_idx]
                # Calculate moving average and handle edges
                ma = trial_data.rolling(window=window, center=True, min_periods=1).mean()
                ma_values.extend(ma.tolist())
            complete_trials_data[f'{vel}_ma{window}'] = ma_values
    
    print("Moving averages calculated.")
    
    # Extract features and targets
    X = complete_trials_data[input_features].values
    y = complete_trials_data[output_features].values
    
    # Scale the data
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    # Fit scalers on training data only
    X_train = X_scaler.fit_transform(X[train_mask])
    y_train = y_scaler.fit_transform(y[train_mask])
    
    # Transform validation and test data
    X_val = X_scaler.transform(X[val_mask])
    y_val = y_scaler.transform(y[val_mask])
    X_test = X_scaler.transform(X[test_mask])
    y_test = y_scaler.transform(y[test_mask])
    
    # Create sequence datasets
    train_dataset = SequenceDataset(X_train, y_train, sequence_length)
    val_dataset = SequenceDataset(X_val, y_val, sequence_length)
    test_dataset = SequenceDataset(X_test, y_test, sequence_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    return (train_loader, val_loader, test_loader), (X_scaler, y_scaler), (input_features, output_features), (train_trials, val_trials, test_trials)

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, patience=20):
    """Train the hybrid model"""
    print("\nTraining model...")
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        
        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            patience_counter = 0
            print(f"  New best model! Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    return best_model, best_val_loss

def evaluate_model(model, test_loader, criterion, device, output_features, 
                  output_scaler, save_dir, test_trials):
    """Evaluate the trained model"""
    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            # Store predictions and targets
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    avg_test_loss = test_loss / len(test_loader)
    print(f"\nTest Loss: {avg_test_loss:.4f}")
    
    # Concatenate all predictions and targets
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    
    # Inverse transform predictions and targets
    predictions = output_scaler.inverse_transform(predictions)
    targets = output_scaler.inverse_transform(targets)
    
    # Calculate metrics for each feature
    metrics = {}
    for i, feature in enumerate(output_features):
        mae = np.mean(np.abs(predictions[:, i] - targets[:, i]))
        rmse = np.sqrt(np.mean((predictions[:, i] - targets[:, i])**2))
        r2 = 1 - np.sum((targets[:, i] - predictions[:, i])**2) / np.sum((targets[:, i] - targets[:, i].mean())**2)
        
        metrics[feature] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2)
        }
        
        # Create prediction plot
        plt.figure(figsize=(10, 6))
        plt.scatter(targets[:, i], predictions[:, i], alpha=0.5)
        plt.plot([targets[:, i].min(), targets[:, i].max()],
                [targets[:, i].min(), targets[:, i].max()],
                'r--', lw=2)
        plt.xlabel(f'Actual {feature}')
        plt.ylabel(f'Predicted {feature}')
        plt.title(f'{feature} Predictions\nMAE = {mae:.3f}, R² = {r2:.3f}')
        plt.grid(True)
        plt.savefig(save_dir / f'{feature}_predictions.png')
        plt.close()
    
    # Save predictions and metadata
    predictions_dir = save_dir / 'predictions'
    predictions_dir.mkdir(exist_ok=True)
    
    np.savez(predictions_dir / 'predictions.npz',
             predictions=predictions,
             targets=targets,
             output_features=output_features,
             trial_indices=test_trials)
    
    # Save metrics
    with open(save_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

def main():
    """Main function to train and evaluate the hybrid model"""
    # Configuration
    config = {
        'data_path': "Z:/Divya/TEMP_transfers/toAni/BPN_P9LT_P9RT_flyCoords.csv",
        'input_features': [
            'x_vel', 'y_vel', 'z_vel',
            'x_vel_ma5', 'y_vel_ma5', 'z_vel_ma5',
            'x_vel_ma10', 'y_vel_ma10', 'z_vel_ma10',
            'x_vel_ma20', 'y_vel_ma20', 'z_vel_ma20'
        ],
        'output_features': [
            'R1A_flex', 'R1A_rot', 'R1A_abduct',
            'R1B_flex', 'R1B_rot',
            'R1C_flex', 'R1C_rot',
            'R1D_flex'
        ],
        'sequence_length': 50,
        'hidden_size': 256,
        'num_lstm_layers': 2,
        'num_transformer_layers': 2,
        'nhead': 8,
        'dropout': 0.1,
        'lstm_ratio': 0.7,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'patience': 20,
        'batch_size': 32,
        'genotype': 'ALL'
    }
    
    # Create directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path('hybrid_results') / f'hybrid_{timestamp}'
    models_dir = results_dir / 'models'
    plots_dir = results_dir / 'plots'
    
    for dir_path in [results_dir, models_dir, plots_dir]:
        dir_path.mkdir(exist_ok=True, parents=True)
    
    # Save configuration
    with open(results_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Prepare data
    (train_loader, val_loader, test_loader), (X_scaler, y_scaler), (input_features, output_features), (train_trials, val_trials, test_trials) = prepare_data(
        config['data_path'],
        config['input_features'],
        config['output_features'],
        config['sequence_length'],
        config['genotype']
    )
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridLSTMTransformer(
        input_size=len(config['input_features']),
        hidden_size=config['hidden_size'],
        output_size=len(config['output_features']),
        num_lstm_layers=config['num_lstm_layers'],
        num_transformer_layers=config['num_transformer_layers'],
        nhead=config['nhead'],
        dropout=config['dropout'],
        lstm_ratio=config['lstm_ratio']
    ).to(device)
    
    # Create loss function and optimizer
    criterion = create_combined_loss(alpha_derivative=0.3)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Train model
    best_model_state, best_val_loss = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        config['num_epochs'], device, config['patience']
    )
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluate model
    metrics = evaluate_model(
        model, test_loader, criterion, device,
        config['output_features'], y_scaler, plots_dir,
        test_trials
    )
    
    # Save model and metadata
    torch.save({
        'model_state_dict': best_model_state,
        'config': config,
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'metrics': metrics,
        'train_trials': train_trials.tolist(),
        'val_trials': val_trials.tolist(),
        'test_trials': test_trials.tolist(),
        'best_val_loss': best_val_loss
    }, models_dir / 'best_model.pt')
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main() 