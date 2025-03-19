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

## Overview

This application provides a comprehensive analysis of HIV patient data using mathematical growth models and machine learning ensemble techniques. It fits multiple models to historical data, evaluates their performance, creates visualizations, and forecasts future HIV trends.

## Features

- Data preprocessing and exploratory analysis
- Implementation of four growth models (Exponential, Logistic, Richards, Gompertz)
- Ensemble modeling using averaging and machine learning approaches
- Time series cross-validation
- Comprehensive metrics evaluation (RMSE, R², MAE)
- Visualization of model fits and forecasts
- Future trend prediction with uncertainty quantification

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
3. Review the generated visualizations and metrics

## Methodology

The project follows these steps:
1. **Data Loading**: Loads and preprocesses HIV patient data
2. **Time Series Preparation**: Creates numerical time indices for modeling
3. **Model Fitting**: Fits individual growth models using curve_fit optimization
4. **Ensemble Building**: Creates ensemble models combining individual models
5. **Evaluation**: Assesses models using RMSE, R², and MAE
6. **Visualization**: Generates plots of model fits, comparisons, and metrics
7. **Forecasting**: Projects future HIV trends with uncertainty bands

## Output Files

The application generates several visualization files:
- `hiv_cases_time_series.png`: Time series plot of historical data
- `hiv_individual_models_comparison.png`: Comparison of individual growth models
- `hiv_ensemble_models_comparison.png`: Comparison of ensemble models
- `hiv_model_metrics_comparison.png`: Visualization of performance metrics
- `hiv_forecast_2024_2028.png`: Future HIV trend forecast (5-year horizon)

## Models Explained

### Growth Models
- **Exponential**: `y = a * exp(b * x) + c` - Models unrestricted growth
- **Logistic**: `y = a / (1 + exp(-b * (x - c))) + d` - Models growth with carrying capacity
- **Richards**: `y = a / (1 + exp(-b * (x - c)))**(1/d) + k` - Generalized logistic with variable growth rate
- **Gompertz**: `y = a * exp(-b * exp(-c * x)) + d` - Asymmetric sigmoid growth

### Ensemble Techniques

- **Simple Average**: Arithmetic mean of all model predictions
- **Weighted Average**: R²-weighted average of model predictions
- **Random Forest**: Decision tree ensemble using model predictions as features
- **Gradient Boosting**: Sequential ensemble using model predictions as features

## Evaluation Metrics

- **RMSE (Root Mean Squared Error)**: Measures average magnitude of prediction errors
- **R² (Coefficient of Determination)**: Indicates proportion of variance explained by the model
- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actual values
