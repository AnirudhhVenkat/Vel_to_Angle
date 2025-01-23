   # Joint Angle Prediction Project

   ## Overview

   This project focuses on predicting joint angles based on time series data, specifically using velocities and accelerations as input features. The project implements various machine learning models, including LSTM, TCN, and Transformer architectures, to analyze and predict the angles effectively.

   ## Features

   - Data preprocessing and preparation
   - Correlation analysis to identify relationships between features and target angles
   - Implementation of multiple machine learning models for prediction
   - Visualization of results through heatmaps and plots

   ## Directory Structure
   
- **new/**: Contains the main project files and subdirectories.
  - **models/**: Implementation of various machine learning models.
    - `base_lstm.py`: Base implementation of an LSTM model.
    - `deep_lstm.py`: Deep LSTM model implementation.
    - `tcn.py`: Temporal Convolutional Network (TCN) model implementation.
    - `transformer.py`: Transformer model implementation.
    - `unsupervised_transformer.py`: Transformer model designed for unsupervised learning.
  - **utils/**: Utility scripts for data handling and performance metrics.
    - `data.py`: Functions for data preprocessing and loading.
    - `metrics.py`: Functions to calculate evaluation metrics.
  - `correlation.py`: Script for analyzing correlations in the dataset.
  - `train.py`: Main script to train and evaluate models.

- **README.md**: Documentation for the project.

   ## Requirements

   To run this project, you need to have the following Python packages installed:

   - pandas
   - numpy
   - matplotlib
   - seaborn
   - torch (PyTorch)
   - scikit-learn
   - tqdm
   - optuna (for hyperparameter optimization)
   - wandb (for experiment tracking)

   You can install the required packages using pip:

   ## Dataset

   The dataset used in this project is located at: Z:/Divya/TEMP_transfers/toAni/BPN_P9LT_P9RT_flyCoords.csv

   ## Usage

   ### Running the Correlation Analysis

   To perform correlation analysis between velocities, accelerations, and angles, run the following command:
python new/correlation.py

### Running the Model Training

To train the LSTM, TCN, and Transformer models, run the following command:
python new/train.py

   This script will handle the training process for the various models implemented in the project.

   ## Contributing

   Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

   ## License

   This project is licensed under the MIT License. See the LICENSE file for more details.

   ## Acknowledgments

   - [PyTorch](https://pytorch.org/) for the deep learning framework.
   - [Optuna](https://optuna.org/) for hyperparameter optimization.
   - [WandB](https://wandb.ai/) for experiment tracking and visualization.
