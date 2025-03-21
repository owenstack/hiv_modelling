# HIV Growth Modeling and Forecasting

This project analyzes HIV patient data from Enugu State, Nigeria (2007-2023) using various growth models and ensemble techniques to understand historical patterns and forecast future trends.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Data Requirements](#data-requirements)
- [Usage](#usage)
- [Methodology](#methodology)
- [Output Files](#output-files)
- [Models Explained](#models-explained)
- [Ensemble Techniques](#ensemble-techniques)
- [Evaluation Metrics](#evaluation-metrics)
- [Visualization](#visualization)

## Overview

This application provides a comprehensive analysis of HIV patient data using mathematical growth models and machine learning ensemble techniques. It fits multiple models to historical data, evaluates their performance using cross-validation, creates detailed visualizations, and forecasts future HIV trends with uncertainty quantification.

## Features

- Data preprocessing and exploratory analysis
- Implementation of four growth models (Exponential, Logistic, Richards, Gompertz)
- Ensemble modeling using simple average, weighted average, and machine learning approaches
- Time series cross-validation for robust evaluation
- Comprehensive metrics evaluation (RMSE, R², MAE)
- Multiple visualization outputs for model comparison and validation
- Future trend prediction with bootstrap uncertainty quantification
- Model persistence for future use

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd models

# Install required packages
pip install -r requirements.txt
```

## Data Requirements

The application expects an Excel file (`data.xlsx`) with the following columns:
- `date`: Dates of observations (will be converted to datetime if needed)
- `number`: Count of HIV patients

## Usage

1. Place your data file in the project directory as `data.xlsx`
2. Run the main script:
   ```bash
   python model.py
   ```
3. Review the generated visualizations in the `plots` directory and the forecast data in `forecast_results_2024_2028.csv`

## Methodology

The project follows these steps:
1. **Data Loading**: Loads and preprocesses HIV patient data
2. **Time Series Preparation**: Creates time indices and splits data for cross-validation
3. **Model Fitting**: Fits individual growth models with improved convergence algorithms
4. **Cross-Validation**: Evaluates models across multiple time-based train/test splits
5. **Ensemble Building**: Creates ensemble models with hyperparameter tuning 
6. **Evaluation**: Assesses models using RMSE, R², and MAE across all CV splits
7. **Visualization**: Generates plots of model fits, comparisons, and performance metrics
8. **Forecasting**: Projects future HIV trends with bootstrap uncertainty bands

## Output Files

The application generates several visualization files in the `plots` directory:
- `hiv_individual_models_comparison.png`: Comparison of individual growth models
- `hiv_ensemble_models_comparison.png`: Comparison of ensemble models
- `hiv_model_metrics_comparison.png`: Visual comparison of performance metrics
- `hiv_model_validation.png`: Validation plot comparing actual vs predicted values
- `hiv_forecast_2024_2028.png`: Future HIV trend forecast (5-year horizon)

Additionally, the forecast data is saved to:
- `forecast_results_2024_2028.csv`: CSV containing predicted values and confidence intervals

## Models Explained

### Growth Models
- **Exponential**: `y = a * exp(b * x) + c` - Models unrestricted growth
- **Logistic**: `y = a / (1 + exp(-b * (x - c))) + d` - Models growth with carrying capacity
- **Richards**: `y = a / (1 + exp(-b * (x - c)))**(1/d) + k` - Generalized logistic with variable growth rate
- **Gompertz**: `y = a * exp(-b * exp(-c * x)) + d` - Asymmetric sigmoid growth

### Ensemble Techniques

- **Simple Average**: Arithmetic mean of all model predictions
- **Weighted Average**: R²-weighted average of model predictions
- **Random Forest**: Decision tree ensemble using model predictions as features with hyperparameter tuning
- **Gradient Boosting**: Sequential ensemble using model predictions as features with hyperparameter tuning

## Evaluation Metrics

- **RMSE (Root Mean Squared Error)**: Measures average magnitude of prediction errors
- **R² (Coefficient of Determination)**: Indicates proportion of variance explained by the model
- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actual values

## Visualization

The system employs a dedicated plot manager to create and save standardized visualizations:
- **Model Fits**: Visualization of how each model fits historical data
- **Ensemble Comparison**: Comparison of ensemble models against individual models
- **Metrics Comparison**: Bar charts showing performance metrics across all models
- **Validation Plot**: Scatterplot of actual vs predicted values to validate model accuracy
- **Future Forecast**: Projection of future trends with confidence intervals
