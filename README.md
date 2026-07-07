# Performance-Prediction-and-Optimization-of-Supercapacitors-Through-Data-Driven-Analysis-
# Machine Learning Based Optimization of Flexible Supercapacitors

## Overview

This project presents a machine learning framework for predicting the capacitance of flexible supercapacitors and recommending optimal material combinations for achieving desired electrochemical performance.

The work was completed as a B.Tech Final Year Project in Electronics and Communication Engineering at IIITDM Kancheepuram.

## Objectives

### 1. Capacitance Prediction

Predict supercapacitor capacitance using material and structural properties such as:

* Specific Surface Area (SSA)
* Pore Volume (PV)
* Pore Width (PW)
* ID/IG Ratio
* Nitrogen Content (N%)
* Oxygen Content (O%)

### 2. Material Recommendation

Recommend suitable:

* Electrode Material
* Electrolyte
* Substrate
* Electrolyte Concentration

for achieving a target:

* Areal Capacitance
* Energy Density
* Power Density

## Datasets

### Capacitance Prediction Dataset

* Original dataset: 121 samples
* Augmented dataset: 1200 samples
* Data augmentation performed using Gaussian noise injection

### Material Prediction Dataset

* 240 supercapacitor configurations
* Features include electrode, electrolyte, substrate, electrolyte concentration and electrochemical performance parameters

## Machine Learning Models

### Capacitance Prediction

* Random Forest Regressor
* XGBoost Regressor
* Support Vector Regression (SVR)
* Multilayer Perceptron (MLP)
* Polynomial Regression
* Stacked Ensemble Model

### Material Recommendation

* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor

## Results

### Capacitance Prediction

| Model         | R² Score |
| ------------- | -------- |
| Stacked Model | 0.95     |
| Random Forest | 0.95     |
| XGBoost       | 0.94     |

Best model: Stacked Ensemble

### Material Recommendation

Best model: Random Forest Regressor

* Accuracy: 99.68%
* R² Score: 0.997

## Repository Structure

* data/ : datasets
* notebooks/ : Jupyter notebooks
* src/ : Python scripts
* figures/ : plots and visualizations
* report/ : final project report
* presentation/ : project presentation

## Author

Ashish Pratap Singh

B.Tech Electronics and Communication Engineering

Indian Institute of Information Technology Design and Manufacturing (IIITDM) Kancheepuram

