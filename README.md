
#  YAW ANGLE OPTIMISATION IN WIND FARMS TO MAXIMISE POWER OUTPUT

## Overview
This repository contains scripts and Jupyter notebooks for processing ChaosBench data, training a Fourier Neural Operator (FNO) model containing a ViT Attention block with physics-informed neural network (PINN), and evaluating its performance. In addition, we present a novel wind farm yaw optimisation framework based on spatiotemporal graph networks (TGN) and surrogate modeling.

## Motivation
Optimiwing wind farm power output is a challenging task due to the complex interaction of wake effects, spatial turbulence, and changing atmospheric conditions. Traditional engineering wake models and static optimisation algorithms often fall short in capturing these dynamics, especially under highly non-stationary wind environments.

Our motivation stems from bridging the gap between large-scale atmospheric forecasting (via ChaosBench) and fine-grained turbine-level optimisation, delivering an end-to-end data-driven yet physics-aware solution.

## Our Novel Approach
We propose a dual-pipeline hybrid framework:
1. A Physics-Constrained FNO-PINN model with ViT inspired attention block for long-range wind forecasting.
2. A TGN-based yaw optimisation model, trained using feedback from a surrogate power model, to dynamically optimise turbine yaw angles over time.

This combination allows us to:
- Forecast atmospheric wind trends using physics-informed learning.
- Exploit the forecast to optimise turbine-level yaw settings via graph-based spatiotemporal reasoning.


## Pipeline 1: Physics-Constrained FNO-PINN

- Input: ERA5 reanalysis variables from ChaosBench.
- Architecture:
  - A FNO backbone for global receptive fields.
  - ViT-inspired Transformer block for feature mixing.
  - PINN losses enforcing divergence-free wind fields and steady-state momentum equations.
- Output: 45-timestep forecasts of key atmospheric variables (wind speed & direction included).

![alt text](https://github.com/R3borN17/alexalowerthetemp/blob/main/Pipeline_1.png?raw=true)

## Pipeline 2: TGN-Based Yaw Optimisation

- Input: Wind forecasts from Pipeline 1.
- Architecture:
  - Graph Convolutional Network (GCN) encodes turbine layout at each timestep.
  - GRU layer processes temporal embeddings across multiple timesteps.
  - Yaw Decoder Head predicts optimal yaw angles for all turbines.
  - A surrogate power model trained on PyWake simulations supervises the learning via power-maximisation objectives.

- Output: Optimised turbine yaw angles for future wind conditions.

![alt text](https://github.com/R3borN17/alexalowerthetemp/blob/main/Pipeline_2.png?raw=true)

## Repository Structure

### Data Processing
- `filter.py`: Filters the ChaosBench dataset to extract only ERA5-based variables for the UK region, preparing data for training.

### Model Training
- `training_fno_pinn.ipynb`: Trains the FNO-PINN model using ChaosBench reanalysis data.
  - Uses:
    - `model.py`: Defines the FNO-PINN architecture with Fourier layers and physics-informed loss.
    - `dataset.py`: Data loader and preprocessing for ERA5 data.
    - `fno_pinn.yaml`: YAML configuration file for hyperparameters.
  - Files located inside `chaosbench-*.zip`.

- `Yaw_angle_Optimisation_Using_Pywake.py` Trains the TGN for dynamic yaw optimisation using Pywake

### Model Evaluation
- `eval_iter.ipynb`: Evaluates the trained FNO-PINN model on test datasets and outputs performance metrics.

### Results
- `loss_curve.png`: Training loss plot.
- `u-velocity_rmse.png` & `v_velocity_rmse.png`: RMSE plots for the wind velocity components (u, v).
- `Predicted_Yaw_Angles_using_Pywake`: Optimised Yaw Angles and Power Output per turbine

## Conclusion
By combining physics-informed forecasting and graph-based yaw optimisation, we achieve a scalable solution for wind farm control. Our hybrid framework dynamically adjusts turbine yaw angles based on realistic wind field forecasts, unlocking power gains beyond static or traditional optimisation techniques.

## How to Use
1. Preprocess Data: Run `filter.py` to extract UK-specific ERA5 data.
2. Train FNO-PINN: Launch `training_fno_pinn.ipynb` with supporting files in the zip archive.
3. Evaluate Model: Use `eval_iter.ipynb` for testing and generating RMSE metrics.
4. Train Surrogate Model: Generate PyWake simulation dataset and train the surrogate model.
5. Train TGN Optimizer: Train the TGN on wind forecasts and surrogate feedback to optimize yaw angles.

## Requirements
```bash
pip install torch numpy matplotlib yaml pytorch-lightning pywake torch-geometric
```

## Acknowledgments
This repository utilises ChaosBench data for model training and evaluation.

Special thanks to Dr. Gege Wen for her constant guidance and support.

Special thanks also to Chunyang Wang for his constant guidance and support.
