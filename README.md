# HIV Growth Modeling and Forecasting

A comprehensive data analysis and forecasting project for HIV enrollment trends in Enugu State, Nigeria (2007-2023).

## Project Overview

This project analyzes historical HIV enrollment data from Enugu State, Nigeria, and builds predictive models to forecast future trends. The analysis includes data cleaning, exploratory data analysis, fitting various growth models, ensemble modeling, and visualization of results.

## Features

- **Data Cleaning & Preprocessing**: Handles missing values, outliers, and creates time-based features
- **Growth Models**: Implements multiple growth models (Exponential, Logistic, Richards, Gompertz)
- **Ensemble Methods**: Combines predictions using simple averaging, weighted averaging, and machine learning models
- **Cross-Validation**: Uses time series cross-validation to ensure robust model evaluation
- **Uncertainty Quantification**: Implements bootstrap methods for confidence intervals
- **Visualization**: Generates comprehensive visualizations for data exploration and model evaluation

## Installation

```bash
# Clone the repository
git clone https://github.com/owenstack/hiv_modeling.git
cd hiv_modeling

# Install dependencies
pip install -r requirements.txt
```

## Data

The project uses HIV enrollment data from Enugu State, Nigeria, spanning from 2007 to 2023. The data includes:

- Daily enrollment numbers
- Cumulative enrollment counts

The raw data is processed to handle missing values, outliers, and to create additional features like:
- Rolling statistics
- Seasonal components
- Time indices

## Methodology

### Data Preprocessing

1. Load and clean the raw data
2. Resample to weekly frequency
3. Add rolling statistics and seasonal features
4. Handle missing values and outliers

### Modeling Approach

The project implements several modeling approaches:

1. **Growth Models**:
   - Exponential: `y = a * exp(b * x) + c`
   - Logistic: `y = a / (1 + exp(-b * (x - c))) + d`
   - Richards: `y = a / (1 + exp(-b * (x - c)))**(1/d) + k`
   - Gompertz: `y = a * exp(-b * exp(-c * x)) + d`

2. **Ensemble Methods**:
   - Simple Average: Equal weighting of all models
   - Weighted Average: Weights based on model performance
   - Random Forest: Tree-based ensemble
   - Gradient Boosting: Boosted tree ensemble

3. **Validation**:
   - Time series cross-validation with 5 splits
   - Performance metrics: RMSE, R², MAE

## Results

Based on the model evaluation, the Logistic growth model performed best among individual models, while the Weighted Average ensemble provided the best overall performance.

### Individual Model Performance

| Model | Avg Test RMSE | Avg Test R² | Avg Test MAE |
|-------|---------------|-------------|--------------|
| Exponential | 7731.71 ± 10163.82 | -1.5141 ± 3.0851 | 7057.25 ± 10077.73 |
| Logistic | 3619.70 ± 3060.53 | 0.2018 ± 1.1377 | 3104.42 ± 2704.13 |
| Richards | 7126.53 ± 10990.52 | -0.8317 ± 3.2696 | 6753.84 ± 10808.15 |
| Gompertz | 7854.88 ± 11119.11 | -1.2094 ± 3.3908 | 7195.67 ± 11024.50 |

### Cross-Validation Results

#### Logistic Model (Best Individual Model)
- Split 1: Test RMSE = 5334.29, R² = -2.0151
- Split 2: Test RMSE = 559.11, R² = 0.9755
- Split 3: Test RMSE = 337.82, R² = 0.9898
- Split 4: Test RMSE = 3379.43, R² = 0.7769
- Split 5: Test RMSE = 8487.82, R² = 0.2818

### Forecasting

The project generates forecasts for HIV enrollment trends from 2024 to 2028, with confidence intervals to quantify uncertainty. The forecast results are saved to `forecast_results_2024_2028.csv`.

## Project Structure

```
├── data/
│   ├── data.xlsx                    # Raw data
│   ├── cleaned_enrollments.csv      # Cleaned data
│   └── forecast_results_2024_2028.csv # Forecast results
├── plots/                           # Generated visualizations
├── saved_models/                    # Saved model files
├── docs/                            # Documentation
├── cleaning.py                      # Data cleaning script
├── model.py                         # Main modeling script
├── visualization.py                 # Visualization functions
├── utilities.py                     # Utility functions
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
```

## Usage

### Data Cleaning

```bash
python cleaning.py
```

This script loads the raw data, performs cleaning operations, and saves the cleaned data to `data/cleaned_enrollments.csv`.

### Model Training and Forecasting

```bash
python model.py
```

This script:
1. Loads the cleaned data
2. Fits various growth models
3. Builds ensemble models
4. Evaluates model performance
5. Generates forecasts
6. Creates visualizations

## Visualizations

The project generates several visualizations:

1. Basic time series plots of the data
2. Individual model fits
3. Ensemble model comparisons
4. Performance metrics comparison
5. Validation plots
6. Forecast plots with confidence intervals

All visualizations are saved to the `plots/` directory.

## Model Tuning

The project includes hyperparameter tuning for the machine learning models:

### Random Forest
Best parameters: {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}

### Gradient Boosting
Best parameters: {'learning_rate': 0.1, 'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300, 'subsample': 1.0}
