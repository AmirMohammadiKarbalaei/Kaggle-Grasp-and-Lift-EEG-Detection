# Grasp-and-Lift EEG Detection

This repository contains notebooks and solutions for the **[Grasp-and-Lift EEG Detection](https://www.kaggle.com/c/grasp-and-lift-eeg-detection)** competition on Kaggle. The goal of this project is to identify when a hand is performing specific kinematic actions (such as grasping, lifting, and replacing an object) from 32-channel continuous EEG signals.

## Project Overview
The task is framed as a **multi-label time-series segmentation problem**. For every timestamp in the EEG signal, the model outputs a probability trace for 6 distinct events:
1. `HandStart`
2. `FirstDigitTouch`
3. `BothStartLoadPhase`
4. `LiftOff`
5. `Replace`
6. `BothReleased`

## Notebooks

### 1. `01_eeg_unet_baseline.ipynb`
A solid baseline demonstrating the end-to-end pipeline from preprocessing to inference.
* **Encoder-Decoder Model**: Uses a 1-D UNet architecture with a pretrained **EfficientNet-B1** encoder (using `timm` and `nnspt`).
* **Loss & Metric**: Trained using Binary Cross Entropy with Logits; evaluated using Macro ROC-AUC.
* **Performance**: Achieves a macro AUC of **~0.824** on the held-out test subject.

### 2. `02_eeg_unet_loso_ensemble.ipynb`
An advanced pipeline designed for robustness and maximizing Kaggle leaderboard performance.
* **Heavier Architecture**: Upgrades the encoder to **EfficientNet-B3**.
* **Data Augmentations**: Incorporates channel dropout, signal scaling, and Gaussian noise to prevent overfitting.
* **LOSO Cross-Validation**: Implements a Leave-One-Subject-Out (LOSO) cross-validation loop to generate 12 robust fold models, rigorously guarding against data leakage.
* **Ensemble & Post-processing**: Averages predictions across all 12 fold models and applies **Causal Smoothing** (rolling mean) to produce the final submission CSV.

## Key Methodologies

### 1. Preprocessing (Wavelet Denoising)
EEG data is notoriously noisy. Both notebooks implement a robust Wavelet Denoising step:
* Wavelet decomposition on each channel (`db2` wavelet, level 3).
* Estimation of noise via Median Absolute Deviation (MAD) of the finest-level coefficients.
* Application of a universal hard threshold before reconstruction.

### 2. Smart Sampling & Soft Labels
Since EEG event onsets are fuzzy (annotations mark a keyframe, not the actual cortical preparation start), the dataset loader uses:
* **Event-Density-Weighted Sampling**: Transition zones and event windows are sampled 6x more often than pure-negative regions. This forces the model to focus on the hardest parts of the signal.
* **Soft Labels**: A label floor of `0.05` is applied to negative samples, preventing the model from becoming over-confident on fuzzy boundaries.

### 3. Training & Validation
* Models process sequences in chunks (e.g., window size of 5000 samples).
* **Standard Scaler** is strictly fitted *only* on the training subjects of the current split to prevent any future/validation data leakage.
* AUC-based early stopping monitors the Macro-AUC across the 6 events to save the best model weights.


### 4. Kaggle Submission
 ![alt text](image.png)
 ![alt text](image-1.png)
 
