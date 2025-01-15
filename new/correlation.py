# correlation.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.data import prepare_data  # Assuming you have a prepare_data function in your data.py

def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def compute_correlation(df):
    """Compute the correlation matrix for the DataFrame."""
    correlation_matrix = df.corr()
    return correlation_matrix

def plot_correlation_heatmap(correlation_matrix):
    """Plot a heatmap of the correlation matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap')
    plt.show()

def main():
    # Specify the path to your dataset
    file_path = "Z:/Divya/TEMP_transfers/toAni/BPN_P9LT_P9RT_flyCoords.csv"  # Your dataset path

    # Load the data
    df = load_data(file_path)
    if df is not None:
        # Prepare the data (if needed)
        # df = prepare_data(config)  # Uncomment if you want to use the prepare_data function

        # Compute the correlation matrix
        correlation_matrix = compute_correlation(df)

        # Print the correlation matrix
        print("Correlation Matrix:")
        print(correlation_matrix)

        # Plot the correlation heatmap
        plot_correlation_heatmap(correlation_matrix)

if __name__ == "__main__":
    main()