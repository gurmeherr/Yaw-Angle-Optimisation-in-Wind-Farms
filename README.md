# ChaosBench Data Filtering and Model Training Repository

## Overview
This repository contains scripts and Jupyter notebooks for processing ChaosBench data, training a Fourier Neural Operator (FNO) model with physics-informed neural networks (PINNs), and evaluating its performance.

## Repository Structure

### Data Processing
- **`filter.py`**: This script filters the ChaosBench dataset to extract only the ERA5 dataset with UK coordinates. This processed data is used for training the model.

### Model Training
- **`training_fno_pinn.ipynb`**: Jupyter notebook for training the FNO-PINN model using the processed dataset. It references:
  - `model.py`: Defines the model architecture.
  - `dataset.py`: Handles data loading and preprocessing.
  - `fno_pinn.yaml`: Configuration file for training arguments and hyperparameters.
  
  These files are located inside the **zip archive (`chaosbench-*.zip`)**.

### Model Evaluation
- **`eval_iter.ipynb`**: Jupyter notebook used for evaluating the trained model on test data.

### Results
- **`loss_curve.png`**: Plot of training loss over epochs.
- **`u-velocity_rmse.png` & `v_velocity_rmse.png`**: RMSE error plots for velocity components.

## How to Use
1. **Preprocess Data**: Run `filter.py` to extract relevant UK-specific ERA5 data.
2. **Train the Model**: Execute `training_fno_pinn.ipynb` while ensuring the required scripts inside the zip file (`model.py`, `dataset.py`, and `fno_pinn.yaml`) are accessible.
3. **Evaluate the Model**: Use `eval_iter.ipynb` to test the trained model and generate performance metrics.
4. **Analyze Results**: Review the output PNG files for training loss and error metrics.

## Requirements
Ensure you have the necessary dependencies installed before running the scripts. You may need:
```bash
pip install torch numpy matplotlib yaml
```

## Acknowledgments
This repository utilizes ChaosBench data for model training and evaluation.

