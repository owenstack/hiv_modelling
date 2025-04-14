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
    forecast_future_trends
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
    """Load and preprocess HIV data from Excel file"""
    try:
        df = pd.read_excel(file_path)
        print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Basic data exploration
        print("\nData Overview:")
        print(df.head())
        print("\nData Information:")
        print(df.info())
        print("\nData Statistics:")
        print(df.describe())
        
        # Check for missing values
        missing_values = df.isnull().sum()
        print("\nMissing Values:")
        print(missing_values)
        
        # Convert date to datetime if not already
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
            print("\nDate column converted to datetime")
        
        # Handle missing cumulative column - compute from 'number' if available
        if 'cumulative' not in df.columns:
            if 'number' in df.columns:
                print("\nComputing cumulative column from 'number' column")
                df = df.sort_values('date')
                df['cumulative'] = df['number'].cumsum()
            else:
                raise ValueError("Neither 'cumulative' nor 'number' column found in dataset")
        
        # Create numerical time variable for modeling
        df['time_idx'] = (df['date'] - df['date'].min()).dt.days
        
        # Sort by date
        df = df.sort_values('date')
        
        # Basic time series plot
        plot_basic_timeseries(df)
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def prepare_data_for_modeling(df, n_splits=5):
    """Prepare data for model fitting with full cross-validation"""
    X = df['time_idx'].values
    y = df['cumulative'].values
    
    # Create TimeSeriesSplit object
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Store all splits for cross-validation
    cv_splits = list(tscv.split(X))
    
    print(f"Prepared {n_splits} time series splits for cross-validation")
    
    return X, y, cv_splits, tscv

# Model Fitting and Evaluation
def fit_growth_models(X, y, cv_splits):
    """Fit individual growth models with cross-validation and improved convergence"""
    models = {
        'Exponential': (exponential_model, ([100, 0.01, 10], [-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])),
        'Logistic': (logistic_model, ([max(y), 0.01, np.median(X), min(y)], 
                                    [0, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf])),
        'Richards': (richards_model, ([max(y), 0.01, np.median(X), 1, min(y)], 
                                    [0, -np.inf, -np.inf, 0.01, -np.inf], [np.inf, np.inf, np.inf, 10, np.inf])),
        'Gompertz': (gompertz_model, ([max(y), 1, 0.01, min(y)], 
                                    [0, 0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]))
    }
    
    fitted_models = {}
    model_metrics = {}
    
    for name, (model_func, bounds) in models.items():
        try:
            print(f"\nFitting {name} Model with cross-validation...")
            
            # Initialize metrics accumulators for cross-validation
            cv_train_rmse = []
            cv_test_rmse = []
            cv_train_r2 = []
            cv_test_r2 = []
            cv_test_mae = []
            best_popt = None
            best_test_r2 = -np.inf
            
            # Loop through all CV splits
            for i, (train_index, test_index) in enumerate(cv_splits):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                # Initialize popt with default bounds
                popt = bounds[0]
                
                # Try different initial guesses if first attempt fails
                for attempt in range(3):
                    try:
                        # Slightly randomize initial parameters for better convergence
                        if attempt > 0:
                            p0 = [p * (1 + 0.1 * np.random.randn()) for p in bounds[0]]
                        else:
                            p0 = bounds[0]
                        
                        # Increased maxfev for better convergence
                        popt, _ = curve_fit(model_func, X_train, y_train, 
                                           bounds=bounds[1:], p0=p0, 
                                           maxfev=50000, method='trf')
                        break
                    except RuntimeError:
                        if attempt == 2:
                            print(f"  Warning: Failed to converge on split {i+1}")
                            popt = bounds[0]  # Use initial guess as fallback
                
                # Predictions
                y_train_pred = model_func(X_train, *popt)
                y_test_pred = model_func(X_test, *popt)
                
                # Metrics
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                
                # Save metrics for this split
                cv_train_rmse.append(train_rmse)
                cv_test_rmse.append(test_rmse)
                cv_train_r2.append(train_r2)
                cv_test_r2.append(test_r2)
                cv_test_mae.append(test_mae)
                
                # Keep track of best parameters
                if test_r2 > best_test_r2:
                    best_test_r2 = test_r2
                    best_popt = popt
                
                print(f"  Split {i+1}: Test RMSE = {test_rmse:.2f}, R² = {test_r2:.4f}")
            
            # Use all data to fit final model with best parameters as initial guess
            final_popt, _ = curve_fit(model_func, X, y, p0=best_popt, 
                                     bounds=bounds[1:], maxfev=50000, method='trf')
            
            # Average metrics across all splits
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
            
        except Exception as e:
            print(f"Error fitting {name} model: {e}")
    
    return fitted_models, model_metrics

# Ensemble Model Building with Hyperparameter Tuning
def build_ensemble_models(X, y, fitted_models, cv_splits):
    """Build ensemble models with hyperparameter tuning"""
    # Create features from model predictions for full dataset
    full_features = np.column_stack([
        model['function'](X, *model['parameters']) 
        for model in fitted_models.values()
    ])
    
    # Initialize metrics accumulators for cross-validation
    ensemble_metrics = {
        'Simple Average': {'train_rmse': [], 'test_rmse': [], 'train_r2': [], 'test_r2': [], 'test_mae': []},
        'Weighted Average': {'train_rmse': [], 'test_rmse': [], 'train_r2': [], 'test_r2': [], 'test_mae': []},
        'Random Forest': {'train_rmse': [], 'test_rmse': [], 'train_r2': [], 'test_r2': [], 'test_mae': []},
        'Gradient Boosting': {'train_rmse': [], 'test_rmse': [], 'train_r2': [], 'test_r2': [], 'test_mae': []}
    }
    
    # Define parameter grids for hyperparameter tuning
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    gb_param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Initialize models and grids
    rf_model = RandomForestRegressor(random_state=42)
    gb_model = GradientBoostingRegressor(random_state=42)
    rf_grid = None
    gb_grid = None
    best_r2_weights = np.ones(len(fitted_models)) / len(fitted_models)
    
    # Process each CV split
    for i, (train_index, test_index) in enumerate(cv_splits):
        # Get data for this split
        X_train_idx, X_test_idx = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Create features for this split
        train_features = np.column_stack([
            model['function'](X_train_idx, *model['parameters']) 
            for model in fitted_models.values()
        ])
        
        test_features = np.column_stack([
            model['function'](X_test_idx, *model['parameters']) 
            for model in fitted_models.values()
        ])
        
        # Simple average ensemble
        simple_avg_train_pred = np.mean(train_features, axis=1)
        simple_avg_test_pred = np.mean(test_features, axis=1)
        
        # Calculate metrics for simple average
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
        
        # Weighted average ensemble (based on R² values from this split)
        # Get R² values for individual models on this test set
        model_r2_values = []
        for name, model in fitted_models.items():
            test_pred = model['function'](X_test_idx, *model['parameters'])
            r2 = float(r2_score(y_test, test_pred))  # Convert to float
            model_r2_values.append(max(0.0, r2))  # Ensure non-negative float
        
        # Normalize R² weights
        if sum(model_r2_values) > 0:
            weights = np.array(model_r2_values) / sum(model_r2_values)
            weighted_avg_train_pred = np.sum(train_features * weights.reshape(1, -1), axis=1)
            weighted_avg_test_pred = np.sum(test_features * weights.reshape(1, -1), axis=1)
            
            # Save weights if this is best split (for future predictions)
            if i == len(cv_splits) - 1:  # Use last split's weights
                best_r2_weights = weights
        else:
            # Fallback to simple average if all R² are zero
            weighted_avg_train_pred = simple_avg_train_pred
            weighted_avg_test_pred = simple_avg_test_pred
            if i == len(cv_splits) - 1:
                best_r2_weights = np.ones(len(fitted_models)) / len(fitted_models)
        
        # Calculate metrics for weighted average
        ensemble_metrics['Weighted Average']['train_rmse'].append(
            np.sqrt(mean_squared_error(y_train, weighted_avg_train_pred)))
        ensemble_metrics['Weighted Average']['test_rmse'].append(
            np.sqrt(mean_squared_error(y_test, weighted_avg_test_pred)))
        ensemble_metrics['Weighted Average']['train_r2'].append(
            r2_score(y_train, weighted_avg_train_pred))
        ensemble_metrics['Weighted Average']['test_r2'].append(
            r2_score(y_test, weighted_avg_test_pred))
        ensemble_metrics['Weighted Average']['test_mae'].append(
            mean_absolute_error(y_test, weighted_avg_test_pred))
        
        # Hyperparameter tuning for ML models - only on first split to save time
        if i == 0:
            # Scale features for better model performance
            scaler = StandardScaler()
            train_features_scaled = scaler.fit_transform(train_features)
            
            # Random Forest with GridSearchCV
            print("\nTuning Random Forest hyperparameters...")
            rf_grid = GridSearchCV(
                rf_model,
                rf_param_grid,
                cv=3,  # Inner CV
                scoring='r2',
                n_jobs=-1
            )
            rf_grid.fit(train_features_scaled, y_train)
            rf_model = rf_grid.best_estimator_
            print(f"Best Random Forest parameters: {rf_grid.best_params_}")
            
            # Gradient Boosting with GridSearchCV
            print("\nTuning Gradient Boosting hyperparameters...")
            gb_grid = GridSearchCV(
                gb_model,
                gb_param_grid,
                cv=3,  # Inner CV
                scoring='r2',
                n_jobs=-1
            )
            gb_grid.fit(train_features_scaled, y_train)
            gb_model = gb_grid.best_estimator_
            print(f"Best Gradient Boosting parameters: {gb_grid.best_params_}")
        
        # Scale test features
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.transform(test_features)
        
        # Use the tuned models for predictions
        rf_train_pred = rf_model.predict(train_features_scaled)
        rf_test_pred = rf_model.predict(test_features_scaled)
        
        gb_train_pred = gb_model.predict(train_features_scaled)
        gb_test_pred = gb_model.predict(test_features_scaled)
        
        # Calculate metrics for Random Forest
        ensemble_metrics['Random Forest']['train_rmse'].append(
            np.sqrt(mean_squared_error(y_train, rf_train_pred)))
        ensemble_metrics['Random Forest']['test_rmse'].append(
            np.sqrt(mean_squared_error(y_test, rf_test_pred)))
        ensemble_metrics['Random Forest']['train_r2'].append(
            r2_score(y_train, rf_train_pred))
        ensemble_metrics['Random Forest']['test_r2'].append(
            r2_score(y_test, rf_test_pred))
        ensemble_metrics['Random Forest']['test_mae'].append(
            mean_absolute_error(y_test, rf_test_pred))
        
        # Calculate metrics for Gradient Boosting
        ensemble_metrics['Gradient Boosting']['train_rmse'].append(
            np.sqrt(mean_squared_error(y_train, gb_train_pred)))
        ensemble_metrics['Gradient Boosting']['test_rmse'].append(
            np.sqrt(mean_squared_error(y_test, gb_test_pred)))
        ensemble_metrics['Gradient Boosting']['train_r2'].append(
            r2_score(y_train, gb_train_pred))
        ensemble_metrics['Gradient Boosting']['test_r2'].append(
            r2_score(y_test, gb_test_pred))
        ensemble_metrics['Gradient Boosting']['test_mae'].append(
            mean_absolute_error(y_test, gb_test_pred))
    
    # Compute average metrics across all splits
    final_metrics = {}
    for model_name, metrics in ensemble_metrics.items():
        final_metrics[model_name] = {
            'train_rmse': np.mean(metrics['train_rmse']),
            'test_rmse': np.mean(metrics['test_rmse']),
            'train_r2': np.mean(metrics['train_r2']),
            'test_r2': np.mean(metrics['test_r2']),
            'test_mae': np.mean(metrics['test_mae']),
            'cv_results': {
                'train_rmse': metrics['train_rmse'],
                'test_rmse': metrics['test_rmse'],
                'train_r2': metrics['train_r2'],
                'test_r2': metrics['test_r2'],
                'test_mae': metrics['test_mae']
            }
        }
    
    # Print ensemble model metrics
    print("\nEnsemble Model Cross-Validation Metrics:")
    for model_name, metrics in final_metrics.items():
        print(f"\n{model_name} Ensemble:")
        print(f"  Avg Train RMSE: {metrics['train_rmse']:.2f} ± {np.std(ensemble_metrics[model_name]['train_rmse']):.2f}")
        print(f"  Avg Test RMSE: {metrics['test_rmse']:.2f} ± {np.std(ensemble_metrics[model_name]['test_rmse']):.2f}")
        print(f"  Avg Train R²: {metrics['train_r2']:.4f} ± {np.std(ensemble_metrics[model_name]['train_r2']):.4f}")
        print(f"  Avg Test R²: {metrics['test_r2']:.4f} ± {np.std(ensemble_metrics[model_name]['test_r2']):.4f}")
        print(f"  Avg Test MAE: {metrics['test_mae']:.2f} ± {np.std(ensemble_metrics[model_name]['test_mae']):.2f}")
    
    # Final models for future predictions
    # Scale full dataset for ML models
    scaler = StandardScaler()
    full_features_scaled = scaler.fit_transform(full_features)
    
    # Train final models on all data
    if rf_grid is not None:
        final_rf_model = RandomForestRegressor(**rf_grid.best_params_, random_state=42)
        final_rf_model.fit(full_features_scaled, y)
    else:
        print("Random Forest hyperparameter tuning failed. Using default parameters.")
        final_rf_model = RandomForestRegressor(random_state=42)
        final_rf_model.fit(full_features_scaled, y)
    
    if gb_grid is not None:
        final_gb_model = GradientBoostingRegressor(**gb_grid.best_params_, random_state=42)
        final_gb_model.fit(full_features_scaled, y)
    else:
        print("Gradient Boosting hyperparameter tuning failed. Using default parameters.")
        final_gb_model = GradientBoostingRegressor(random_state=42)
        final_gb_model.fit(full_features_scaled, y)
    
    # Save the RF and GB models
    os.makedirs('saved_models', exist_ok=True)
    joblib.dump(final_rf_model, 'saved_models/rf_model.pkl')
    joblib.dump(final_gb_model, 'saved_models/gb_model.pkl')
    joblib.dump(scaler, 'saved_models/feature_scaler.pkl')
    
    ensemble_models = {
        'Simple Average': {
            'predict': lambda x: np.mean(x, axis=1)
        },
        'Weighted Average': {
            'predict': lambda x: np.sum(x * best_r2_weights.reshape(1, -1), axis=1),
            'weights': best_r2_weights
        },
        'Random Forest': {
            'model': final_rf_model,
            'scaler': scaler
        },
        'Gradient Boosting': {
            'model': final_gb_model,
            'scaler': scaler
        }
    }
    
    return ensemble_models, final_metrics

# Main function
def main():
    # Load and explore data
    file_path = 'data.xlsx'
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
    
    # Visualize individual model fits
    visualize_model_fits(df, X, y, final_train_idx, final_test_idx, y_train, y_test, fitted_models)
    
    # Build ensemble models with hyperparameter tuning
    ensemble_models, ensemble_metrics = build_ensemble_models(X, y, fitted_models, cv_splits)
    
    # Identify best models
    best_model_name = max(model_metrics, key=lambda k: model_metrics[k]['test_r2'])
    best_ensemble_model_name = max(ensemble_metrics, key=lambda k: ensemble_metrics[k]['test_r2'])
    
    print(f"\nBest Individual Model: {best_model_name}")
    print(f"Best Ensemble Model: {best_ensemble_model_name}")
    
    # Visualize ensemble comparisons
    visualize_ensemble_comparison(df, X, y, cv_splits, fitted_models, ensemble_models, best_model_name)
    
    # Compare model metrics
    visualize_metrics_comparison(model_metrics, ensemble_metrics)
    
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
    
    print("\nAnalysis complete. All visualizations saved to disk.")
    print(f"Forecast data saved to forecast_results_2024_2028.csv")

if __name__ == "__main__":
    main()
