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

| Model         | Avg Test RMSE   | Avg Test R²     | Avg Test MAE    |
|---------------|-----------------|-----------------|-----------------|
| Exponential   | 7731.71         | -1.5141         | 7057.25         |
| Logistic      | 3619.70         | 0.2018          | 3104.42         |
| Richards      | 7126.53         | -0.8317         | 6753.84         |
| Gompertz      | 7854.88         | -1.2094         | 7195.67         |

### Cross-Validation Results

#### Exponential Model Cross-Validation Metrics

  Avg Train RMSE: 486.60 ± 437.89
  Avg Test RMSE: 7731.71 ± 10163.81
  Avg Train R²: 0.9970 ± 0.0023
  Avg Test R²: -1.5141 ± 3.0851
  Avg Test MAE: 7057.25 ± 10077.72

#### Logistic Model Cross-Validation Metrics

  Avg Train RMSE: 1100.78 ± 1500.00
  Avg Test RMSE: 4075.72 ± 3808.04
  Avg Train R²: 0.9871 ± 0.0207
  Avg Test R²: 0.1150 ± 1.1438
  Avg Test MAE: 3568.53 ± 3543.17

#### Richards Model Cross-Validation Metrics

  Avg Train RMSE: 527.61 ± 496.69
  Avg Test RMSE: 7892.04 ± 10770.11
  Avg Train R²: 0.9968 ± 0.0025
  Avg Test R²: -1.1786 ± 3.1786
  Avg Test MAE: 7123.10 ± 10679.06

#### Gompertz Model Cross-Validation Metrics

  Avg Train RMSE: 498.60 ± 491.00
  Avg Test RMSE: 7854.88 ± 11119.11
  Avg Train R²: 0.9971 ± 0.0025
  Avg Test R²: -1.2094 ± 3.3908
  Avg Test MAE: 7195.67 ± 11024.50

### Ensemble Input Tables

#### Weighted Average Ensemble Input Table (Head)

| time_idx | Exponential_pred | Logistic_pred | Richards_pred | Gompertz_pred | Weighted_Average_pred |
|---|---|---|---|---|---|
| 0 | 4409.282432 | 6612.948670 | 6409.763572 | 8083.998099 | 6281.605931 |
| 1 | 4437.766884 | 6629.995000 | 6427.049953 | 8087.920546 | 6299.030342 |
| 2 | 4466.330301 | 6647.115536 | 6444.411198 | 8091.898150 | 6316.526486 |
| 3 | 4494.972899 | 6664.310587 | 6461.847616 | 8095.931528 | 6334.094712 |
| 4 | 4523.694901 | 6681.580466 | 6479.359521 | 8100.021301 | 6351.735372 |

#### Random Forest Ensemble Input Table (Head)

| time_idx | Exponential_pred | Logistic_pred | Richards_pred | Gompertz_pred | Random_Forest_pred |
|---|---|---|---|---|---|
| 0 | 4409.282432 | 6612.948670 | 6409.763572 | 8083.998099 | 806.580 |
| 1 | 4437.766884 | 6629.995000 | 6427.049953 | 8087.920546 | 48.280 |
| 2 | 4466.330301 | 6647.115536 | 6444.411198 | 8091.898150 | 63.100 |
| 3 | 4494.972899 | 6664.310587 | 6461.847616 | 8095.931528 | 79.005 |
| 4 | 4523.694901 | 6681.580466 | 6479.359521 | 8100.021301 | 104.355 |

#### Gradient Boosting Ensemble Input Table (Head)

| time_idx | Exponential_pred | Logistic_pred | Richards_pred | Gompertz_pred | Gradient_Boosting_pred |
|---|---|---|---|---|---|
| 0 | 4409.282432 | 6612.948670 | 6409.763572 | 8083.998099 | 32.362315 |
| 1 | 4437.766884 | 6629.995000 | 6427.049953 | 8087.920546 | 51.234939 |
| 2 | 4466.330301 | 6647.115536 | 6444.411198 | 8091.898150 | 68.755695 |
| 3 | 4494.972899 | 6664.310587 | 6461.847616 | 8095.931528 | 81.178686 |
| 4 | 4523.694901 | 6681.580466 | 6479.359521 | 8100.021301 | 109.677626 |

### Summary Statistics of Predictions

| | mean | std | min | max |
|---|---|---|---|---|
| Actual_Cumulative_Cases | 38647.799324 | 30462.940980 | 27.000000 | 105782.000000 |
| Exponential_pred | 38647.799337 | 30150.797036 | 4409.282432 | 113533.808947 |
| Logistic_pred | 38647.799347 | 29869.696022 | 6612.948670 | 108778.053080 |
| Richards_pred | 38647.799273 | 30009.536473 | 6409.763572 | 109031.087072 |
| Gompertz_pred | 38647.799319 | 29495.450942 | 8083.998099 | 103103.545169 |
| Simple_Average_pred | 38647.799319 | 29866.687782 | 6378.998193 | 108611.623567 |
| Weighted_Average_pred | 38647.799319 | 29891.022016 | 6281.605931 | 108931.871662 |
| Random_Forest_pred | 38630.511195 | 30450.117185 | 48.280000 | 105761.595000 |
| Gradient_Boosting_pred | 38647.799324 | 30462.852459 | 32.362315 | 105776.701399 |

### Forecasting

The project generates forecasts for HIV enrollment trends from 2024 to 2028, with confidence intervals to quantify uncertainty. The forecast results are saved to `data/forecast_results_2024_2028.csv`.

## Project Structure

``` chart
├── data/
│   ├── data.xlsx                    # Raw data
│   ├── cleaned_enrollments.csv      # Cleaned data
│   └── forecast_results_2024_2028.csv # Forecast results
├── plots/                           # Generated visualizations
├── saved_models/                    # Saved model files
├── tables/                          # Generated data tables
│   ├── full_predictions_comparison.csv
│   └── predictions_summary_statistics.csv
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
