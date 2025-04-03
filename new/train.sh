#!/bin/bash

# This script runs the LSTM and LSTM-Transformer training scripts
# To make executable: chmod +x train.sh

#SBATCH --partition=shortq7
#SBATCH -N 1
#SBATCH --mem-per-cpu=16000
#SBATCH --job-name=train_lstm_transformer
#SBATCH --output=train_lstm_transformer.out
#SBATCH --error=train_lstm_transformer.err

# Set error handling
set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as an error when substituting

# Log start time
echo "=== Training started at $(date) ==="

# Navigate to script directory (in case the script is called from elsewhere)
cd "$(dirname "$0")"

echo "=== Starting LSTM-Transformer training ==="
python train_lstm_transformer.py
if [ $? -eq 0 ]; then
    echo "✓ LSTM-Transformer training completed successfully"
else
    echo "✗ LSTM-Transformer training failed with error code $?"
    exit 1
fi

echo "=== Starting LSTM training ==="
python train_lstm.py
if [ $? -eq 0 ]; then
    echo "✓ LSTM training completed successfully"
else
    echo "✗ LSTM training failed with error code $?"
    exit 1
fi

# Log completion
echo "=== All training completed successfully at $(date) ==="

