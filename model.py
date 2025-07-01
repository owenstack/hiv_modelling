import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib
import os
from functools import partial
import warnings

from visualization import (
    plot_basic_timeseries,
    visualize_model_fits,
    visualize_ensemble_comparison,
    visualize_metrics_comparison,
    create_validation_plot,
    forecast_future_trends,
    plot_residuals,
    plot_qq,
    plot_residuals_histogram,
    plot_individual_model_fit
)
from utilities import generate_bootstrap_predictions

warnings.filterwarnings('ignore')

# Growth Models
def exponential_model(x, a, b, c):
    """Exponential growth model: y = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c

def logistic_model(x, a, b, c, d):
    """Logistic growth model: y = a / (1 + np.exp(-b * (x - c))) + d"""
    return a / (1 + np.exp(-b * (x - c))) + d

def richards_model(x, a, b, c, d, k):
    """Richards growth model: y = a / (1 + exp(-b * (x - c)))**(1/d) + k"""
    return a / (1 + np.exp(-b * (x - c)))**(1/d) + k

def gompertz_model(x, a, b, c, d):
    """Gompertz growth model: y = a * exp(-b * exp(-c * x)) + d"""
    return a * np.exp(-b * np.exp(-c * x)) + d

# Data Loading and Preprocessing
def load_data(file_path):
    """Load and preprocess HIV data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Keep only relevant columns
        df = df[['date', 'number', 'cumulative']]
        
        # Convert date to datetime if not already
        df['date'] = pd.to_datetime(df['date'])
        
        # Resample to weekly data
        df.set_index('date', inplace=True)
        weekly_df = df.resample('W').agg({
            'number': 'sum',
            'cumulative': 'last'
        }).reset_index()
        
        # Add rolling statistics
        weekly_df['rolling_mean_4w'] = weekly_df['number'].rolling(window=4).mean()
        weekly_df['rolling_mean_12w'] = weekly_df['number'].rolling(window=12).mean()
        
        # Add seasonal features
        weekly_df['month'] = weekly_df['date'].dt.month
        weekly_df['quarter'] = weekly_df['date'].dt.quarter
        
        # Create numerical time variable for modeling
        weekly_df['time_idx'] = (weekly_df['date'] - weekly_df['date'].min()).dt.days / 7  # in weeks
        
        # Forward fill any NaN from rolling calculations
        weekly_df.fillna(method='ffill', inplace=True)
        
        # Basic time series plot
        plot_basic_timeseries(weekly_df)
        
        print("\nData resampled to weekly frequency with added features")
        print(f"New shape: {weekly_df.shape}")
        
        return weekly_df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def prepare_data_for_modeling(df, n_splits=5):
    """Prepare data for model fitting with full cross-validation"""
    # Extract features
    X = df['time_idx'].values
    y = df['cumulative'].values
    
    # Validate data
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("Invalid values detected in target variable")
    
    if len(y) < 52:  # At least 1 year of weekly data
        raise ValueError("Insufficient data points for modeling")
    
    # Ensure strictly increasing cumulative numbers
    y = np.maximum.accumulate(y)
    
    # Calculate minimum split size to ensure enough data for training
    min_samples = len(y) // (n_splits + 1)
    if min_samples < 26:  # At least 6 months of weekly data per split
        n_splits = max(2, len(y) // 26 - 1)
        print(f"Adjusted n_splits to {n_splits} to ensure sufficient data per split")
    
    # Create TimeSeriesSplit object with gap
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=4)  # 4-week gap between train and test
    
    # Store all splits for cross-validation
    cv_splits = list(tscv.split(X))
    
    print(f"\nPrepared {n_splits} time series splits for cross-validation")
    print(f"Data points per split: ~{min_samples}")
    
    return X, y, cv_splits, tscv

# Model Fitting and Evaluation
def fit_growth_models(X, y, cv_splits):
    """Fit individual growth models with cross-validation and improved convergence"""
    models = {
        'Exponential': {
            'func': exponential_model,
            'bounds': ([np.max(y)/2, 0.005, min(y)], [0, 0.0001, -np.inf], [np.max(y)*2, 0.1, np.inf]),
            'param_names': ['a', 'b', 'c']
        },
        'Logistic': {
            'func': logistic_model,
            'bounds': ([np.max(y)*1.2, 0.01, np.median(X), min(y)], [np.max(y), 0.001, X.min(), -np.inf], [np.max(y)*2, 0.1, X.max(), np.inf]),
            'param_names': ['a', 'b', 'c', 'd']
        },
        'Richards': {
            'func': richards_model,
            'bounds': ([np.max(y)*1.2, 0.01, np.median(X), 1, min(y)], [np.max(y), 0.001, X.min(), 0.1, -np.inf], [np.max(y)*2, 0.1, X.max(), 10, np.inf]),
            'param_names': ['a', 'b', 'c', 'd', 'k']
        },
        'Gompertz': {
            'func': gompertz_model,
            'bounds': ([np.max(y)*1.2, 2, 0.01, min(y)], [np.max(y), 0.1, 0.001, -np.inf], [np.max(y)*2, 10, 0.1, np.inf]),
            'param_names': ['a', 'b', 'c', 'd']
        }
    }
    
    fitted_models = {}
    model_metrics = {}
    
    for name, model_details in models.items():
        model_func = model_details['func']
        bounds = model_details['bounds']
        param_names = model_details['param_names']

        try:
            print(f"\nFitting {name} Model with cross-validation...")
            
            # Initialize metrics accumulators
            cv_train_rmse = []
            cv_test_rmse = []
            cv_train_r2 = []
            cv_test_r2 = []
            cv_test_mae = []
            best_popt = None
            best_test_r2 = -np.inf
            
            # First fit on full data to get better initial parameters
            try:
                full_popt, _ = curve_fit(model_func, X, y, 
                                       p0=bounds[0],
                                       bounds=(bounds[1], bounds[2]),
                                       maxfev=100000,
                                       method='trf')
                initial_params = full_popt
            except:
                initial_params = bounds[0]
            
            # Loop through CV splits with improved fitting strategy
            for i, (train_index, test_index) in enumerate(cv_splits):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                best_split_popt = None
                best_split_r2 = -np.inf
                
                # Try multiple initializations
                for attempt in range(5):
                    try:
                        if attempt == 0:
                            p0 = initial_params
                        else:
                            # Random perturbation of parameters
                            noise = np.random.normal(0, 0.1 * (attempt / 5), len(initial_params))
                            p0 = initial_params * (1 + noise)
                            p0 = np.clip(p0, bounds[1], bounds[2])
                        
                        popt, _ = curve_fit(model_func, X_train, y_train,
                                          p0=p0,
                                          bounds=(bounds[1], bounds[2]),
                                          maxfev=50000,
                                          method='trf')
                        
                        # Check if this attempt is better
                        y_test_pred = model_func(X_test, *popt)
                        test_r2 = r2_score(y_test, y_test_pred)
                        
                        if test_r2 > best_split_r2:
                            best_split_r2 = test_r2
                            best_split_popt = popt
                            
                    except RuntimeError:
                        continue
                
                if best_split_popt is None:
                    print(f"  Warning: Failed to converge on split {i+1}")
                    best_split_popt = initial_params
                
                # Calculate metrics using best parameters for this split
                y_train_pred = model_func(X_train, *best_split_popt)
                y_test_pred = model_func(X_test, *best_split_popt)
                
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                
                # Store metrics
                cv_train_rmse.append(train_rmse)
                cv_test_rmse.append(test_rmse)
                cv_train_r2.append(train_r2)
                cv_test_r2.append(test_r2)
                cv_test_mae.append(test_mae)
                
                if test_r2 > best_test_r2:
                    best_test_r2 = test_r2
                    best_popt = best_split_popt
                
                print(f"  Split {i+1}: Test RMSE = {test_rmse:.2f}, R² = {test_r2:.4f}")
            
            # Final fit using best parameters as initialization
            final_popt, _ = curve_fit(model_func, X, y,
                                    p0=best_popt,
                                    bounds=(bounds[1], bounds[2]),
                                    maxfev=100000,
                                    method='trf')
            
            # Generate predictions on the full dataset for diagnostic plots
            y_pred_full = model_func(X, *final_popt)
            residuals_full = y - y_pred_full

            # Generate diagnostic plots
            plot_residuals(y, y_pred_full, name, filename_suffix="_full_data",
                           filename=f"residuals_{name}_full_data.png")
            plot_qq(residuals_full, name, filename_suffix="_full_data",
                    filename=f"qq_plot_{name}_full_data.png")
            plot_residuals_histogram(residuals_full, name, filename_suffix="_full_data",
                                     filename=f"residuals_histogram_{name}_full_data.png")

            # Store results
            model_metrics[name] = {
                'train_rmse': np.mean(cv_train_rmse),
                'test_rmse': np.mean(cv_test_rmse),
                'train_r2': np.mean(cv_train_r2),
                'test_r2': np.mean(cv_test_r2),
                'test_mae': np.mean(cv_test_mae),
                'parameters': final_popt,
                'cv_results': {
                    'train_rmse': cv_train_rmse,
                    'test_rmse': cv_test_rmse,
                    'train_r2': cv_train_r2,
                    'test_r2': cv_test_r2,
                    'test_mae': cv_test_mae
                }
            }
            
            fitted_models[name] = {
                'function': model_func,
                'parameters': final_popt
            }
            
            print(f"{name} Model Cross-Validation Metrics:")
            print(f"  Avg Train RMSE: {np.mean(cv_train_rmse):.2f} ± {np.std(cv_train_rmse):.2f}")
            print(f"  Avg Test RMSE: {np.mean(cv_test_rmse):.2f} ± {np.std(cv_test_rmse):.2f}")
            print(f"  Avg Train R²: {np.mean(cv_train_r2):.4f} ± {np.std(cv_train_r2):.4f}")
            print(f"  Avg Test R²: {np.mean(cv_test_r2):.4f} ± {np.std(cv_test_r2):.4f}")
            print(f"  Avg Test MAE: {np.mean(cv_test_mae):.2f} ± {np.std(cv_test_mae):.2f}")

            print(f"\n{name} Model - Final Estimated Parameters:")
            param_table_header = "Parameter | Value"
            print(param_table_header)
            print("-" * len(param_table_header))
            for p_name, p_val in zip(param_names, final_popt):
                print(f"{p_name:<9} | {p_val:.4f}")
            
        except Exception as e:
            print(f"Error fitting {name} model: {e}")
    
    return fitted_models, model_metrics

# Ensemble Model Building with Improved Feature Engineering and Weight Calculation
def build_ensemble_models(X, y, fitted_models, cv_splits):
    """Build ensemble models with improved feature engineering and weighting"""
    # Create base features from model predictions
    base_features = np.column_stack([
        model['function'](X, *model['parameters']) 
        for model in fitted_models.values()
    ])
    
    # Add engineered features
    time_idx_norm = (X - X.min()) / (X.max() - X.min())
    feature_matrix = np.column_stack([
        base_features,
        time_idx_norm,
        np.sin(2 * np.pi * time_idx_norm),  # Seasonal components
        np.cos(2 * np.pi * time_idx_norm)
    ])
    
    # Initialize metrics accumulators
    ensemble_metrics = {
        'Simple Average': {'train_rmse': [], 'test_rmse': [], 'train_r2': [], 'test_r2': [], 'test_mae': []},
        'Weighted Average (ExpR2)': {'train_rmse': [], 'test_rmse': [], 'train_r2': [], 'test_r2': [], 'test_mae': [], 'weights_history': []},
        'Weighted Average (TestR2)': {'train_rmse': [], 'test_rmse': [], 'train_r2': [], 'test_r2': [], 'test_mae': [], 'weights_history': []},
        'Weighted Average (InvTestMSE)': {'train_rmse': [], 'test_rmse': [], 'train_r2': [], 'test_r2': [], 'test_mae': [], 'weights_history': []},
        'Random Forest': {'train_rmse': [], 'test_rmse': [], 'train_r2': [], 'test_r2': [], 'test_mae': []},
        'Gradient Boosting': {'train_rmse': [], 'test_rmse': [], 'train_r2': [], 'test_r2': [], 'test_mae': []}
    }
    
    # To store the final weights for each scheme (e.g., from the last fold or averaged)
    final_weights = {
        'ExpR2': None,
        'TestR2': None,
        'InvTestMSE': None
    }

    # Enhanced parameter grids
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2', None]
    }
    
    gb_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # Initialize models
    rf_model = RandomForestRegressor(random_state=42)
    gb_model = GradientBoostingRegressor(random_state=42)
    rf_grid = None
    gb_grid = None
    best_r2_weights = np.ones(base_features.shape[1]) / base_features.shape[1]
    
    # Process each CV split with improved weighting
    for i, (train_index, test_index) in enumerate(cv_splits):
        X_train_idx, X_test_idx = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Create features for this split
        train_time_idx_norm = (X_train_idx - X.min()) / (X.max() - X.min())
        test_time_idx_norm = (X_test_idx - X.min()) / (X.max() - X.min())
        
        train_base_features = np.column_stack([
            model['function'](X_train_idx, *model['parameters']) 
            for model in fitted_models.values()
        ])
        test_base_features = np.column_stack([
            model['function'](X_test_idx, *model['parameters']) 
            for model in fitted_models.values()
        ])
        
        train_features = np.column_stack([
            train_base_features,
            train_time_idx_norm,
            np.sin(2 * np.pi * train_time_idx_norm),
            np.cos(2 * np.pi * train_time_idx_norm)
        ])
        test_features = np.column_stack([
            test_base_features,
            test_time_idx_norm,
            np.sin(2 * np.pi * test_time_idx_norm),
            np.cos(2 * np.pi * test_time_idx_norm)
        ])
        
        # Simple average (base models only)
        simple_avg_train_pred = np.mean(train_base_features, axis=1)
        simple_avg_test_pred = np.mean(test_base_features, axis=1)
        
        # Calculate and store metrics for simple average
        ensemble_metrics['Simple Average']['train_rmse'].append(
            np.sqrt(mean_squared_error(y_train, simple_avg_train_pred)))
        ensemble_metrics['Simple Average']['test_rmse'].append(
            np.sqrt(mean_squared_error(y_test, simple_avg_test_pred)))
        ensemble_metrics['Simple Average']['train_r2'].append(
            r2_score(y_train, simple_avg_train_pred))
        ensemble_metrics['Simple Average']['test_r2'].append(
            r2_score(y_test, simple_avg_test_pred))
        ensemble_metrics['Simple Average']['test_mae'].append(
            mean_absolute_error(y_test, simple_avg_test_pred))
        
        # --- Weighted Average Schemes ---
        epsilon = 1e-9 # For numerical stability

        # Scheme 1: Exponential R² (Existing)
        weights_expr2 = []
        for j in range(train_base_features.shape[1]):
            train_r2_val = float(max(0.0, r2_score(y_train, train_base_features[:, j])))
            test_r2_val = float(max(0.0, r2_score(y_test, test_base_features[:, j])))
            weights_expr2.append(np.exp(train_r2_val + test_r2_val))
        weights_expr2 = np.array(weights_expr2)
        if sum(weights_expr2) > epsilon:
            weights_expr2 = weights_expr2 / sum(weights_expr2)
        else:
            weights_expr2 = np.ones(base_features.shape[1]) / base_features.shape[1] # Fallback to equal weights
        ensemble_metrics['Weighted Average (ExpR2)']['weights_history'].append(weights_expr2)

        # Scheme 2: Test R² based
        weights_test_r2 = []
        for j in range(test_base_features.shape[1]):
            r2_val = float(max(0.0, r2_score(y_test, test_base_features[:, j]))) # Use only test R²
            weights_test_r2.append(r2_val)
        weights_test_r2 = np.array(weights_test_r2)
        if sum(weights_test_r2) > epsilon:
            weights_test_r2 = weights_test_r2 / sum(weights_test_r2)
        else:
            weights_test_r2 = np.ones(base_features.shape[1]) / base_features.shape[1]
        ensemble_metrics['Weighted Average (TestR2)']['weights_history'].append(weights_test_r2)

        # Scheme 3: Inverse Test MSE based
        weights_inv_mse = []
        for j in range(test_base_features.shape[1]):
            mse_val = mean_squared_error(y_test, test_base_features[:, j])
            weights_inv_mse.append(1.0 / (mse_val + epsilon))
        weights_inv_mse = np.array(weights_inv_mse)
        if sum(weights_inv_mse) > epsilon:
            weights_inv_mse = weights_inv_mse / sum(weights_inv_mse)
        else:
            weights_inv_mse = np.ones(base_features.shape[1]) / base_features.shape[1]
        ensemble_metrics['Weighted Average (InvTestMSE)']['weights_history'].append(weights_inv_mse)

        # Calculate predictions and metrics for each weighted average scheme
        for scheme_name, current_weights_list in [
            ('Weighted Average (ExpR2)', ensemble_metrics['Weighted Average (ExpR2)']['weights_history'][-1]),
            ('Weighted Average (TestR2)', ensemble_metrics['Weighted Average (TestR2)']['weights_history'][-1]),
            ('Weighted Average (InvTestMSE)', ensemble_metrics['Weighted Average (InvTestMSE)']['weights_history'][-1])
        ]:
            wa_train_pred = np.sum(train_base_features * current_weights_list.reshape(1, -1), axis=1)
            wa_test_pred = np.sum(test_base_features * current_weights_list.reshape(1, -1), axis=1)

            ensemble_metrics[scheme_name]['train_rmse'].append(np.sqrt(mean_squared_error(y_train, wa_train_pred)))
            ensemble_metrics[scheme_name]['test_rmse'].append(np.sqrt(mean_squared_error(y_test, wa_test_pred)))
            ensemble_metrics[scheme_name]['train_r2'].append(r2_score(y_train, wa_train_pred))
            ensemble_metrics[scheme_name]['test_r2'].append(r2_score(y_test, wa_test_pred))
            ensemble_metrics[scheme_name]['test_mae'].append(mean_absolute_error(y_test, wa_test_pred))
        
        # Store weights from the last fold as the "final" weights for later use (e.g. for full data prediction)
        if i == len(cv_splits) - 1:
            final_weights['ExpR2'] = weights_expr2
            final_weights['TestR2'] = weights_test_r2
            final_weights['InvTestMSE'] = weights_inv_mse
        
        # Scale features for ML models
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.transform(test_features)
        
        # Hyperparameter tuning on first split
        if i == 0:
            print("\nTuning Random Forest hyperparameters...")
            rf_grid = GridSearchCV(
                rf_model,
                rf_param_grid,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            rf_grid.fit(train_features_scaled, y_train)
            rf_model = rf_grid.best_estimator_
            print(f"Best Random Forest parameters: {rf_grid.best_params_}")
            
            print("\nTuning Gradient Boosting hyperparameters...")
            gb_grid = GridSearchCV(
                gb_model,
                gb_param_grid,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            gb_grid.fit(train_features_scaled, y_train)
            gb_model = gb_grid.best_estimator_
            print(f"Best Gradient Boosting parameters: {gb_grid.best_params_}")
        
        # Get predictions from ML models
        rf_train_pred = rf_model.predict(train_features_scaled)
        rf_test_pred = rf_model.predict(test_features_scaled)
        gb_train_pred = gb_model.predict(train_features_scaled)
        gb_test_pred = gb_model.predict(test_features_scaled)
        
        # Store ML model metrics
        for name, train_pred, test_pred in [('Random Forest', rf_train_pred, rf_test_pred),
                                          ('Gradient Boosting', gb_train_pred, gb_test_pred)]:
            ensemble_metrics[name]['train_rmse'].append(
                np.sqrt(mean_squared_error(y_train, train_pred)))
            ensemble_metrics[name]['test_rmse'].append(
                np.sqrt(mean_squared_error(y_test, test_pred)))
            ensemble_metrics[name]['train_r2'].append(
                r2_score(y_train, train_pred))
            ensemble_metrics[name]['test_r2'].append(
                r2_score(y_test, test_pred))
            ensemble_metrics[name]['test_mae'].append(
                mean_absolute_error(y_test, test_pred))
    
    # Compute final metrics and prepare models for deployment
    final_metrics = {name: {
        metric: np.mean(values[metric])
        for metric in ['train_rmse', 'test_rmse', 'train_r2', 'test_r2', 'test_mae']
    } for name, values in ensemble_metrics.items()}
    
    # Scale full dataset and train final models
    scaler = StandardScaler()
    full_features_scaled = scaler.fit_transform(feature_matrix)
    
    if rf_grid and gb_grid:
        final_rf = RandomForestRegressor(**rf_grid.best_params_, random_state=42)
        final_gb = GradientBoostingRegressor(**gb_grid.best_params_, random_state=42)
    else:
        final_rf = RandomForestRegressor(n_estimators=200, random_state=42)
        final_gb = GradientBoostingRegressor(n_estimators=200, random_state=42)
    
    final_rf.fit(full_features_scaled, y)
    final_gb.fit(full_features_scaled, y)
    
    # Save models
    os.makedirs('saved_models', exist_ok=True)
    joblib.dump(final_rf, 'saved_models/rf_model.pkl')
    joblib.dump(final_gb, 'saved_models/gb_model.pkl')
    joblib.dump(scaler, 'saved_models/feature_scaler.pkl')

    # --- Generate and Save Input Tables for Ensemble Techniques ---
    os.makedirs('tables', exist_ok=True)

    # Base predictions from individual models on the full dataset
    individual_model_names = list(fitted_models.keys())
    df_individual_preds = pd.DataFrame(base_features, columns=[f"{name}_pred" for name in individual_model_names])
    df_individual_preds.insert(0, 'time_idx', X)

    # Weighted Average Ensemble Input Table
    weighted_avg_full_pred = np.sum(base_features * best_r2_weights.reshape(1, -1), axis=1)
    df_weighted_avg_input = df_individual_preds.copy()
    df_weighted_avg_input['Weighted_Average_pred'] = weighted_avg_full_pred
    print("\n--- Weighted Average Ensemble Input Table (Head) ---")
    print(df_weighted_avg_input.head())
    df_weighted_avg_input.to_csv('tables/ensemble_input_weighted_average_predictions.csv', index=False)

    # Random Forest Ensemble Input Table
    rf_full_pred = final_rf.predict(full_features_scaled)
    df_rf_input = df_individual_preds.copy()
    df_rf_input['Random_Forest_pred'] = rf_full_pred
    print("\n--- Random Forest Ensemble Input Table (Head) ---")
    print(df_rf_input.head())
    df_rf_input.to_csv('tables/ensemble_input_random_forest_predictions.csv', index=False)

    # Gradient Boosting Ensemble Input Table
    gb_full_pred = final_gb.predict(full_features_scaled)
    df_gb_input = df_individual_preds.copy()
    df_gb_input['Gradient_Boosting_pred'] = gb_full_pred
    print("\n--- Gradient Boosting Ensemble Input Table (Head) ---")
    print(df_gb_input.head())
    df_gb_input.to_csv('tables/ensemble_input_gradient_boosting_predictions.csv', index=False)
    # --- End of Input Tables Generation ---
    
    # Create prediction functions that handle feature engineering
    def create_features(X_new_tdi): # Changed X_new to X_new_tdi to avoid conflict with outer scope X
        # This function relies on 'fitted_models' and outer scope 'X' (for min/max norm) from build_ensemble_models
        # Ensure this context is correct when this function is eventually called.
        base_f = np.column_stack([
            model['function'](X_new_tdi, *model['parameters'])
            for model in fitted_models.values()
        ])
        # Normalize X_new_tdi using the original X's min/max for consistency
        time_idx_norm_new = (X_new_tdi - X.min()) / (X.max() - X.min() + 1e-9)
        return np.column_stack([
            base_f,
            time_idx_norm_new,
            np.sin(2 * np.pi * time_idx_norm_new),
            np.cos(2 * np.pi * time_idx_norm_new)
        ])
    
    # If CV didn't run (e.g. n_splits too low or error), final_weights might not be set.
    # Fallback to equal weights if any scheme's weights are missing.
    num_base_models = base_features.shape[1]
    for scheme in ['ExpR2', 'TestR2', 'InvTestMSE']:
        if final_weights[scheme] is None:
            print(f"Warning: final_weights for {scheme} not set, falling back to equal weights.")
            final_weights[scheme] = np.ones(num_base_models) / num_base_models

    ensemble_models = {
        'Simple Average': {
            'predict': lambda bf: np.mean(bf[:, :num_base_models], axis=1)
        },
        'Weighted Average (ExpR2)': {
            'predict': lambda bf: np.sum(bf[:, :num_base_models] * final_weights['ExpR2'].reshape(1, -1), axis=1),
            'weights': final_weights['ExpR2']
        },
        'Weighted Average (TestR2)': {
            'predict': lambda bf: np.sum(bf[:, :num_base_models] * final_weights['TestR2'].reshape(1, -1), axis=1),
            'weights': final_weights['TestR2']
        },
        'Weighted Average (InvTestMSE)': {
            'predict': lambda bf: np.sum(bf[:, :num_base_models] * final_weights['InvTestMSE'].reshape(1, -1), axis=1),
            'weights': final_weights['InvTestMSE']
        },
        'Random Forest': {
            'model': final_rf,
            'scaler': scaler,
            'create_features_func': create_features # Store the function itself
        },
        'Gradient Boosting': {
            'model': final_gb,
            'scaler': scaler,
            'create_features_func': create_features # Store the function itself
        }
    }
    
    return ensemble_models, final_metrics, final_weights # Return final_weights as well
# Main function
def main():
    # Load and explore data
    file_path = 'data/cleaned_enrollments.csv'
    df = load_data(file_path)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Prepare data for modeling with cross-validation
    X, y, cv_splits, tscv = prepare_data_for_modeling(df, n_splits=5)
    
    # Use last split for visualization
    final_train_idx, final_test_idx = cv_splits[-1]
    X_train, X_test = X[final_train_idx], X[final_test_idx]
    y_train, y_test = y[final_train_idx], y[final_test_idx]
    
    # Fit individual growth models
    global fitted_models, model_metrics, best_model_name
    fitted_models, model_metrics = fit_growth_models(X, y, cv_splits)

    # Display Performance Metrics Table for Individual Models
    print("\n--- Individual Model Performance Metrics ---")
    metrics_to_display = ['test_rmse', 'test_r2', 'test_mae']
    header = f"{'Model':<15} | {'Avg Test RMSE':<15} | {'Avg Test R²':<15} | {'Avg Test MAE':<15}"
    print(header)
    print("-" * len(header))
    for model_name, metrics in model_metrics.items():
        print(f"{model_name:<15} | {metrics['test_rmse']:<15.2f} | {metrics['test_r2']:<15.4f} | {metrics['test_mae']:<15.2f}")
    print("--------------------------------------------")

    # Visualize individual model fits (combined plot)
    visualize_model_fits(df, X, y, final_train_idx, final_test_idx, y_train, y_test, fitted_models)

    # Generate and save individual Observed vs. Predicted charts for each model
    print("\nGenerating individual Observed vs. Predicted charts...")
    for model_name, model_data in fitted_models.items():
        plot_individual_model_fit(
            df_date_col=df['date'],
            X_time_idx=X,
            y_actual=y,
            train_indices=final_train_idx, # Using last CV split for representative train/test visual
            test_indices=final_test_idx,
            model_name=model_name,
            model_function=model_data['function'],
            model_parameters=model_data['parameters'],
            filename=f"fit_observed_vs_predicted_{model_name}.png"
        )
    print("Individual Observed vs. Predicted charts saved to 'plots/' directory.")
    
    # Build ensemble models with hyperparameter tuning
    # Now also returns final_weights for the weighted average schemes
    ensemble_models, ensemble_metrics, final_ensemble_weights = build_ensemble_models(X, y, fitted_models, cv_splits)
    
    # Identify best models
    best_model_name = max(model_metrics, key=lambda k: model_metrics[k]['test_r2'])
    best_ensemble_model_name = max(ensemble_metrics, key=lambda k: ensemble_metrics[k]['test_r2'])
    
    print(f"\nBest Individual Model: {best_model_name}")
    print(f"Best Ensemble Model: {best_ensemble_model_name}")
    
    # Visualize ensemble comparisons
    visualize_ensemble_comparison(df, X, y, cv_splits, fitted_models, ensemble_models, best_model_name)
    
    # Compare model metrics (overall)
    visualize_metrics_comparison(model_metrics, ensemble_metrics)

    # --- Compare Weighted Average Schemes ---
    print("\n--- Weighted Average Scheme Performance Comparison ---")
    wa_scheme_names = ['Weighted Average (ExpR2)', 'Weighted Average (TestR2)', 'Weighted Average (InvTestMSE)']
    wa_comparison_data = {
        'Scheme': [],
        'Avg Test RMSE': [],
        'Avg Test R²': [],
        'Avg Test MAE': []
    }
    for scheme_name in wa_scheme_names:
        if scheme_name in ensemble_metrics:
            metrics = ensemble_metrics[scheme_name]
            wa_comparison_data['Scheme'].append(scheme_name)
            wa_comparison_data['Avg Test RMSE'].append(metrics.get('test_rmse', np.nan)) # Use .get for safety
            wa_comparison_data['Avg Test R²'].append(metrics.get('test_r2', np.nan))
            wa_comparison_data['Avg Test MAE'].append(metrics.get('test_mae', np.nan))
        else:
            print(f"Warning: Metrics for {scheme_name} not found in ensemble_metrics.")

    if wa_comparison_data['Scheme']: # Check if any data was added
        df_wa_comparison = pd.DataFrame(wa_comparison_data)
        print(df_wa_comparison.to_string(index=False))
        df_wa_comparison.to_csv('tables/weighted_average_scheme_comparison.csv', index=False)
        print("Weighted Average scheme comparison table saved to 'tables/weighted_average_scheme_comparison.csv'")
    else:
        print("No data to compare for Weighted Average schemes.")
    # --- End of Weighted Average Scheme Comparison ---

    # --- Generate Detailed Parameter, Weight, and Growth Rate Tables ---
    print("\n--- Generating Detailed Weighted Average Scheme Tables ---")
    growth_rate_param_indices = {
        'Exponential': 1, # 'b'
        'Logistic': 1,    # 'b'
        'Richards': 1,    # 'b'
        'Gompertz': 2     # 'c'
    }
    individual_model_names = list(fitted_models.keys()) # Should be ['Exponential', 'Logistic', 'Richards', 'Gompertz']

    for scheme_key, scheme_name_full in [
        ('ExpR2', 'Weighted Average (ExpR2)'),
        ('TestR2', 'Weighted Average (TestR2)'),
        ('InvTestMSE', 'Weighted Average (InvTestMSE)')
    ]:
        if scheme_key not in final_ensemble_weights or final_ensemble_weights[scheme_key] is None:
            print(f"Skipping detailed table for {scheme_name_full} as weights are not available.")
            continue

        current_weights = final_ensemble_weights[scheme_key]
        table_data = {
            'Individual Model': [],
            'Growth_Rate_Parameter_Value': [],
            f'Assigned_Weight_{scheme_key}': []
        }
        total_weighted_growth_rate = 0

        for i, model_name in enumerate(individual_model_names):
            table_data['Individual Model'].append(model_name)

            param_idx = growth_rate_param_indices[model_name]
            growth_rate_val = fitted_models[model_name]['parameters'][param_idx]
            table_data['Growth_Rate_Parameter_Value'].append(growth_rate_val)

            weight_val = current_weights[i]
            table_data[f'Assigned_Weight_{scheme_key}'].append(weight_val)

            total_weighted_growth_rate += growth_rate_val * weight_val

        df_scheme_details = pd.DataFrame(table_data)
        print(f"\n--- Details for {scheme_name_full} ---")
        print(df_scheme_details.to_string(index=False))
        print(f"Overall Weighted Growth Rate for {scheme_name_full}: {total_weighted_growth_rate:.4f}")

        df_scheme_details.to_csv(f'tables/weighted_average_details_{scheme_key}.csv', index=False)
        print(f"Detailed table for {scheme_name_full} saved to 'tables/weighted_average_details_{scheme_key}.csv'")
    # --- End of Detailed Parameter, Weight, and Growth Rate Tables ---
    
    # Create validation plot
    create_validation_plot(
        df, X, y,
        fitted_models[best_model_name],
        ensemble_models[best_ensemble_model_name],
        best_ensemble_model_name,
        fitted_models,
        best_model_name
    )
    
    # Forecast future trends with enhanced uncertainty quantification
    forecast_future_trends(
        df, X, y,
        fitted_models[best_model_name],
        ensemble_models[best_ensemble_model_name],
        best_ensemble_model_name,
        fitted_models,
        best_model_name,
        generate_bootstrap_predictions
    )
    
    # --- Generate and Display Full Prediction Table (Overall Comparison) ---
    print("\n--- Generating Full Prediction Table ---")
    full_pred_df = pd.DataFrame({
        'time_idx': X,
        'date': df['date'],
        'Actual_Cumulative_Cases': y
    })

    # Add individual model predictions
    for model_name, model_data in fitted_models.items():
        full_pred_df[f"{model_name}_pred"] = model_data['function'](X, *model_data['parameters'])

    # Add ensemble model predictions
    # Base features for simple and weighted average
    base_features_full = np.column_stack([
        model['function'](X, *model['parameters'])
        for model in fitted_models.values()
    ])

    # Simple Average
    full_pred_df['Simple_Average_pred'] = ensemble_models['Simple Average']['predict'](base_features_full)

    # Weighted Average Schemes
    full_pred_df['Weighted_Average_ExpR2_pred'] = ensemble_models['Weighted Average (ExpR2)']['predict'](base_features_full)
    full_pred_df['Weighted_Average_TestR2_pred'] = ensemble_models['Weighted Average (TestR2)']['predict'](base_features_full)
    full_pred_df['Weighted_Average_InvTestMSE_pred'] = ensemble_models['Weighted Average (InvTestMSE)']['predict'](base_features_full)

    # Random Forest & Gradient Boosting
    rf_model_info = ensemble_models['Random Forest']
    gb_model_info = ensemble_models['Gradient Boosting']

    # Create features for RF and GB using the function stored in ensemble_models
    # This requires X (time_idx) and access to individual fitted_models to reconstruct base_features within create_features
    # The 'create_features' function in build_ensemble_models needs to be robust or adjusted if X context is different
    # For simplicity, we assume 'create_features' correctly uses the global X and fitted_models context
    # or that it's self-contained if fitted_models is passed appropriately.
    # The current 'create_features' is defined within build_ensemble_models and uses 'fitted_models' from its scope.
    # To call it here, we might need to redefine it or make it accessible with the correct context.

    # Re-creating features for RF/GB based on how it's done in build_ensemble_models
    # This is a bit redundant but ensures consistency if 'create_features' func from dict is tricky to call directly here.
    time_idx_norm_full = (X - X.min()) / (X.max() - X.min() + 1e-9) # Add epsilon to avoid division by zero if X is constant
    rf_gb_features_full = np.column_stack([
        base_features_full,
        time_idx_norm_full,
        np.sin(2 * np.pi * time_idx_norm_full),
        np.cos(2 * np.pi * time_idx_norm_full)
    ])

    rf_scaled_features_full = rf_model_info['scaler'].transform(rf_gb_features_full)
    full_pred_df['Random_Forest_pred'] = rf_model_info['model'].predict(rf_scaled_features_full)

    gb_scaled_features_full = gb_model_info['scaler'].transform(rf_gb_features_full) # Should be the same scaler if trained together
    full_pred_df['Gradient_Boosting_pred'] = gb_model_info['model'].predict(gb_scaled_features_full)

    print("\n--- Full Prediction Table (Head) ---")
    print(full_pred_df.head())
    print("\n--- Full Prediction Table (Tail) ---")
    print(full_pred_df.tail())
    full_pred_df.to_csv('tables/full_predictions_comparison.csv', index=False)
    print("Full prediction table saved to 'tables/full_predictions_comparison.csv'")
    # --- End of Full Prediction Table ---

    # --- Generate and Display Summary Statistics Table of Full Predictions ---
    print("\n--- Generating Summary Statistics of Predictions ---")

    # Select columns for summary: Actuals and all prediction columns
    summary_cols = ['Actual_Cumulative_Cases'] + [col for col in full_pred_df.columns if '_pred' in col]
    summary_stats_df = full_pred_df[summary_cols].agg(['mean', 'std', 'min', 'max']).transpose()

    # Rename columns for clarity if needed, though default 'mean', 'std', 'min', 'max' are fine.
    # summary_stats_df.columns = ['Mean', 'Standard Deviation', 'Minimum', 'Maximum']

    print("\n--- Summary Statistics of Predictions ---")
    print(summary_stats_df)
    summary_stats_df.to_csv('tables/predictions_summary_statistics.csv', index=True) # index=True to keep model names as row index
    print("Summary statistics table saved to 'tables/predictions_summary_statistics.csv'")
    # --- End of Summary Statistics Table ---

    print("\nAnalysis complete. All visualizations and tables saved to disk.")
    print(f"Forecast data saved to data/forecast_results_2024_2028.csv")

if __name__ == "__main__":
    main()
