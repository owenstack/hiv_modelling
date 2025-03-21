import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
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
warnings.filterwarnings('ignore')

# Set styling for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')

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

def plot_basic_timeseries(df):
    """Create basic time series plot of the data"""
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['cumulative'], marker='o', linestyle='-', alpha=0.7)
    plt.title('Cumulative HIV Cases Over Time in Enugu State, Nigeria (2007-2023)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Number of HIV Patients')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/hiv_cumulative_cases_time_series.png')
    plt.close()

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
    
    return ensemble_models, final_metrics# Visualization Functions
def plot_manager(plot_func):
    """Decorator for plot functions to handle saving and closing plots"""
    def wrapper(*args, **kwargs):
        fig = plot_func(*args, **kwargs)
        
        # Create plots directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        
        if 'filename' in kwargs:
            plt.savefig(os.path.join('plots', kwargs['filename']))
        else:
            plt.savefig(os.path.join('plots', f"{plot_func.__name__}.png"))
        plt.close(fig)
        return fig
    return wrapper

@plot_manager
def visualize_model_fits(df, X, y, X_train, X_test, y_train, y_test, fitted_models, filename='hiv_individual_models_comparison.png'):
    """Visualize individual model fits"""
    fig = plt.figure(figsize=(16, 10))
    
    # Plot actual data
    plt.scatter(df['date'].iloc[X_train], y_train, color='blue', alpha=0.6, label='Training Data')
    plt.scatter(df['date'].iloc[X_test], y_test, color='red', alpha=0.6, label='Testing Data')
    
    # Predict for full date range and plot fitted models
    colors = ['green', 'purple', 'orange', 'brown']
    for i, (name, model) in enumerate(fitted_models.items()):
        y_pred = model['function'](X, *model['parameters'])
        plt.plot(df['date'], y_pred, color=colors[i % len(colors)], linewidth=2, label=f'{name} Model')
    
    plt.title('Cumulative HIV Growth Models Comparison - Enugu State, Nigeria (2007-2023)', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative Number of HIV Patients', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

@plot_manager
def visualize_ensemble_comparison(df, X, y, cv_splits, fitted_models, ensemble_models, filename='hiv_ensemble_models_comparison.png'):
    """Visualize and compare ensemble models performance"""
    fig = plt.figure(figsize=(16, 10))
    
    # Use the last split for visualization
    train_idx, test_idx = cv_splits[-1]
    
    # Plot actual data
    plt.scatter(df['date'].iloc[train_idx], y[train_idx], color='blue', alpha=0.6, label='Training Data')
    plt.scatter(df['date'].iloc[test_idx], y[test_idx], color='red', alpha=0.6, label='Testing Data')
    
    # Plot best individual model
    best_model_name = max(model_metrics, key=lambda k: model_metrics[k]['test_r2'])
    best_model = fitted_models[best_model_name]
    y_pred_best = best_model['function'](X, *best_model['parameters'])
    plt.plot(df['date'], y_pred_best, color='green', linewidth=2, label=f'Best Individual ({best_model_name})')
    
    # Create features for full range prediction
    full_features = np.column_stack([
        model['function'](X, *model['parameters']) 
        for model in fitted_models.values()
    ])
    
    # Plot ensemble models
    colors = ['purple', 'orange', 'brown', 'magenta']
    ensemble_models_to_plot = ['Simple Average', 'Weighted Average', 'Random Forest', 'Gradient Boosting']
    
    for i, name in enumerate(ensemble_models_to_plot):
        if name in ['Simple Average', 'Weighted Average']:
            y_pred = ensemble_models[name]['predict'](full_features)
        else:  # Machine learning ensembles
            scaled_features = ensemble_models[name]['scaler'].transform(full_features)
            y_pred = ensemble_models[name]['model'].predict(scaled_features)
            
        plt.plot(df['date'], y_pred, color=colors[i], linewidth=2, label=f'{name} Ensemble')
    
    plt.title('HIV Growth Ensemble Models Comparison - Enugu State, Nigeria (2007-2023)', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Number of HIV Patients', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

@plot_manager
def visualize_metrics_comparison(model_metrics, ensemble_metrics, filename='hiv_model_metrics_comparison.png'):
    """Visualize performance metrics comparison between models"""
    # Combine all metrics
    all_metrics = {**model_metrics, **ensemble_metrics}
    
    # Create DataFrames for easier plotting
    metrics_data = {
        'Model': [],
        'Type': [],
        'RMSE': [],
        'R²': [],
        'MAE': []
    }
    
    for model_name, metrics in all_metrics.items():
        metrics_data['Model'].append(model_name)
        metrics_data['Type'].append('Individual' if model_name in model_metrics else 'Ensemble')
        metrics_data['RMSE'].append(metrics['test_rmse'])
        metrics_data['R²'].append(metrics['test_r2'])
        metrics_data['MAE'].append(metrics['test_mae'])
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create visualizations
    fig = plt.figure(figsize=(18, 15))
    gs = GridSpec(2, 2, figure=fig)
    
    # RMSE comparison
    ax1 = fig.add_subplot(gs[0, 0])
    sns.barplot(x='Model', y='RMSE', hue='Type', data=metrics_df, ax=ax1)
    ax1.set_title('Root Mean Squared Error (RMSE) Comparison', fontsize=14)
    ax1.set_xlabel('')
    ax1.set_ylabel('RMSE (lower is better)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # R² comparison
    ax2 = fig.add_subplot(gs[0, 1])
    sns.barplot(x='Model', y='R²', hue='Type', data=metrics_df, ax=ax2)
    ax2.set_title('R² Score Comparison', fontsize=14)
    ax2.set_xlabel('')
    ax2.set_ylabel('R² (higher is better)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    # MAE comparison
    ax3 = fig.add_subplot(gs[1, 0])
    sns.barplot(x='Model', y='MAE', hue='Type', data=metrics_df, ax=ax3)
    ax3.set_title('Mean Absolute Error (MAE) Comparison', fontsize=14)
    ax3.set_xlabel('')
    ax3.set_ylabel('MAE (lower is better)', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    
    # Overall metric ranking
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate a combined score (normalized)
    metrics_df['RMSE_norm'] = (metrics_df['RMSE'] - metrics_df['RMSE'].min()) / (metrics_df['RMSE'].max() - metrics_df['RMSE'].min() + 1e-10)
    metrics_df['R²_norm'] = 1 - (metrics_df['R²'] - metrics_df['R²'].min()) / (metrics_df['R²'].max() - metrics_df['R²'].min() + 1e-10)
    metrics_df['MAE_norm'] = (metrics_df['MAE'] - metrics_df['MAE'].min()) / (metrics_df['MAE'].max() - metrics_df['MAE'].min() + 1e-10)
    
    metrics_df['Overall_Score'] = (1 - metrics_df['RMSE_norm'] + metrics_df['R²_norm'] + (1 - metrics_df['MAE_norm'])) / 3
    
    # Sort by overall score
    metrics_df = metrics_df.sort_values('Overall_Score', ascending=False)
    
    sns.barplot(x='Model', y='Overall_Score', hue='Type', data=metrics_df, ax=ax4)
    ax4.set_title('Overall Model Performance Score', fontsize=14)
    ax4.set_xlabel('')
    ax4.set_ylabel('Performance Score (higher is better)', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    return fig

def generate_bootstrap_predictions(model_func, X, params, n_samples=100, confidence=0.95):
    """Generate bootstrap predictions with confidence intervals"""
    # Slight parameter perturbations for each bootstrap
    bootstrap_predictions = np.zeros((n_samples, len(X)))
    
    for i in range(n_samples):
        # Perturb parameters slightly for each bootstrap sample
        perturbed_params = [p * (1 + 0.05 * np.random.randn()) for p in params]
        bootstrap_predictions[i] = model_func(X, *perturbed_params)
    
    # Calculate median and confidence intervals
    lower_quantile = (1 - confidence) / 2
    upper_quantile = 1 - lower_quantile
    
    median_predictions = np.median(bootstrap_predictions, axis=0)
    lower_bound = np.quantile(bootstrap_predictions, lower_quantile, axis=0)
    upper_bound = np.quantile(bootstrap_predictions, upper_quantile, axis=0)
    
    return median_predictions, lower_bound, upper_bound

@plot_manager
def forecast_future_trends(df, X, y, best_model, best_ensemble_model, ensemble_name, fitted_models, filename='hiv_forecast_2024_2028.png'):
    """Forecast future HIV trends with enhanced uncertainty quantification"""
    # Create future time points (next 5 years)
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=5*365, freq='D')
    
    # Create time index for future dates
    max_time_idx = df['time_idx'].max()
    future_time_idx = np.arange(max_time_idx + 1, max_time_idx + 1 + len(future_dates))
    
    # Bootstrap predictions for uncertainty quantification
    _, lower_bound, upper_bound = generate_bootstrap_predictions(
        best_model['function'], future_time_idx, best_model['parameters'], n_samples=1000
    )
    
    # Predict using best individual model
    best_model_future = best_model['function'](future_time_idx, *best_model['parameters'])
    
    # Generate features from individual models for ensemble predictions
    future_features = np.column_stack([
        model['function'](future_time_idx, *model['parameters']) 
        for model in fitted_models.values()
    ])
    
    # Predict using best ensemble model based on ensemble type
    if ensemble_name in ['Random Forest', 'Gradient Boosting']:
        # ML-based ensemble models
        scaled_features = best_ensemble_model['scaler'].transform(future_features)
        best_ensemble_future = best_ensemble_model['model'].predict(scaled_features)
    else:
        # Average-based ensembles
        best_ensemble_future = best_ensemble_model['predict'](future_features)
    
    # Create future DataFrame
    future_df = pd.DataFrame({
        'date': future_dates,
        'time_idx': future_time_idx,
        'best_model_forecast': best_model_future,
        'best_ensemble_forecast': best_ensemble_future,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    })
    
    # Create plot
    fig = plt.figure(figsize=(16, 8))
    
    # Historical data
    plt.scatter(df['date'], y, color='blue', alpha=0.5, label='Historical Data')
    
    # Historical fitted values
    historical_fitted = best_model['function'](X, *best_model['parameters'])
    plt.plot(df['date'], historical_fitted, color='green', linestyle='-', linewidth=2, 
             label=f'Best Individual Model Fit ({best_model_name})')
    
    # Future forecasts
    plt.plot(future_df['date'], future_df['best_model_forecast'], color='green', linestyle='--', 
             linewidth=2, label=f'Best Individual Model Forecast ({best_model_name})')
    plt.plot(future_df['date'], future_df['best_ensemble_forecast'], color='purple', linestyle='--', 
             linewidth=2, label=f'Best Ensemble Forecast ({ensemble_name})')
    
    # Add enhanced forecast uncertainty bands
    plt.fill_between(future_df['date'], 
                     future_df['lower_bound'],
                     future_df['upper_bound'],
                     color='gray', alpha=0.3, label='95% Confidence Interval')
    
    plt.title('Cumulative HIV Trend Forecast (2024-2028) - Enugu State, Nigeria', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative Number of HIV Patients', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save forecast data for further analysis
    future_df.to_csv('forecast_results_2024_2028.csv', index=False)
    
    return fig

@plot_manager
def create_validation_plot(df, X, y, best_model, best_ensemble_model, ensemble_name, fitted_models, filename='hiv_model_validation.png'):
    """Create validation plot comparing actual vs predicted values for historical data"""
    # Generate predictions for historical period
    historical_best_model = best_model['function'](X, *best_model['parameters'])
    
    # Generate features from individual models for historical period
    historical_features = np.column_stack([
        model['function'](X, *model['parameters']) 
        for model in fitted_models.values()
    ])
    
    # Get historical ensemble predictions based on ensemble type
    if ensemble_name in ['Random Forest', 'Gradient Boosting']:
        scaled_features = best_ensemble_model['scaler'].transform(historical_features)
        historical_ensemble = best_ensemble_model['model'].predict(scaled_features)
    else:
        historical_ensemble = best_ensemble_model['predict'](historical_features)
    
    # Create plot
    fig = plt.figure(figsize=(16, 8))
    
    # Plot actual data
    plt.plot(df['date'], y, color='blue', linewidth=2, label='Actual Cumulative HIV Cases')
    
    # Plot predictions
    plt.plot(df['date'], historical_best_model, color='green', linestyle='--', 
             linewidth=2, label=f'Best Individual Model ({best_model_name})')
    plt.plot(df['date'], historical_ensemble, color='purple', linestyle='--', 
             linewidth=2, label=f'Best Ensemble Model ({ensemble_name})')
    
    # Calculate errors
    individual_rmse = np.sqrt(mean_squared_error(y, historical_best_model))
    ensemble_rmse = np.sqrt(mean_squared_error(y, historical_ensemble))
    individual_mae = mean_absolute_error(y, historical_best_model)
    ensemble_mae = mean_absolute_error(y, historical_ensemble)
    
    plt.title('Validation: Actual vs Predicted HIV Cases (2007-2023)', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative Number of HIV Patients', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
    plt.xticks(rotation=45)
    
    # Add error metrics to plot
    plt.figtext(0.15, 0.15, f"Best Individual Model RMSE: {individual_rmse:.2f}\nMAE: {individual_mae:.2f}",
                bbox=dict(facecolor='white', alpha=0.8))
    plt.figtext(0.15, 0.05, f"Best Ensemble Model RMSE: {ensemble_rmse:.2f}\nMAE: {ensemble_mae:.2f}",
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    return fig

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
    global fitted_models, model_metrics
    fitted_models, model_metrics = fit_growth_models(X, y, cv_splits)
    
    # Visualize individual model fits
    visualize_model_fits(df, X, y, final_train_idx, final_test_idx, y_train, y_test, fitted_models)
    
    # Build ensemble models with hyperparameter tuning
    ensemble_models, ensemble_metrics = build_ensemble_models(X, y, fitted_models, cv_splits)
    
    # Visualize ensemble comparisons
    visualize_ensemble_comparison(df, X, y, cv_splits, fitted_models, ensemble_models)
    
    # Compare model metrics
    visualize_metrics_comparison(model_metrics, ensemble_metrics)
    
    # Identify best models
    global best_model_name
    best_model_name = max(model_metrics, key=lambda k: model_metrics[k]['test_r2'])
    best_ensemble_model_name = max(ensemble_metrics, key=lambda k: ensemble_metrics[k]['test_r2'])
    
    print(f"\nBest Individual Model: {best_model_name}")
    print(f"Best Ensemble Model: {best_ensemble_model_name}")
    
    # Forecast future trends with enhanced uncertainty quantification
    future_df = forecast_future_trends(
        df, X, y, 
        fitted_models[best_model_name],
        ensemble_models[best_ensemble_model_name],
        best_ensemble_model_name,
        fitted_models
    )
    
    # Create validation plot
    create_validation_plot(
        df, X, y,
        fitted_models[best_model_name],
        ensemble_models[best_ensemble_model_name],
        best_ensemble_model_name,
        fitted_models
    )
    
    print("\nAnalysis complete. All visualizations saved to disk.")
    print(f"Forecast data saved to forecast_results_2024_2028.csv")

if __name__ == "__main__":
    main()
