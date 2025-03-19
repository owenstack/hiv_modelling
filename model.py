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
from sklearn.model_selection import train_test_split, TimeSeriesSplit
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
        
        # Create numerical time variable for modeling
        df['time_idx'] = (df['date'] - df['date'].min()).dt.days
        
        # Sort by date
        df = df.sort_values('date')
        
        # Check for seasonality and trends
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['number'], marker='o', linestyle='-', alpha=0.7)
        plt.title('HIV Cases Over Time in Enugu State, Nigeria (2007-2023)')
        plt.xlabel('Date')
        plt.ylabel('Number of HIV Patients')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('hiv_cases_time_series.png')
        plt.close()
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def prepare_data_for_modeling(df):
    """Prepare data for model fitting"""
    X = df['time_idx'].values
    y = df['number'].values
    
    # Initialize variables
    X_train = None
    X_test = None
    y_train = None 
    y_test = None
    
    # Create training and testing splits with time consideration
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        break  # Just take the first split for simplicity
    
    # Verify splits were created
    if X_train is None or X_test is None or y_train is None or y_test is None:
        raise ValueError("Failed to create train/test splits")
        
    return X, y, X_train, X_test, y_train, y_test
# Model Fitting and Evaluation
def fit_growth_models(X_train, y_train, X_test, y_test):
    """Fit individual growth models and evaluate performance"""
    models = {
        'Exponential': (exponential_model, ([100, 0.01, 10], [-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])),
        'Logistic': (logistic_model, ([max(y_train), 0.01, np.median(X_train), min(y_train)], 
                                    [0, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf])),
        'Richards': (richards_model, ([max(y_train), 0.01, np.median(X_train), 1, min(y_train)], 
                                    [0, -np.inf, -np.inf, 0.01, -np.inf], [np.inf, np.inf, np.inf, 10, np.inf])),
        'Gompertz': (gompertz_model, ([max(y_train), 1, 0.01, min(y_train)], 
                                    [0, 0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]))
    }
    
    fitted_models = {}
    model_metrics = {}
    
    for name, (model_func, bounds) in models.items():
        try:
            print(f"\nFitting {name} Model...")
            popt, _ = curve_fit(model_func, X_train, y_train, bounds=bounds[1:], p0=bounds[0], maxfev=10000)
            
            # Predictions
            y_train_pred = model_func(X_train, *popt)
            y_test_pred = model_func(X_test, *popt)
            
            # Metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            model_metrics[name] = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'parameters': popt
            }
            
            fitted_models[name] = {
                'function': model_func,
                'parameters': popt
            }
            
            print(f"{name} Model Metrics:")
            print(f"  Train RMSE: {train_rmse:.2f}")
            print(f"  Test RMSE: {test_rmse:.2f}")
            print(f"  Train R²: {train_r2:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  Test MAE: {test_mae:.2f}")
            
        except Exception as e:
            print(f"Error fitting {name} model: {e}")
    
    return fitted_models, model_metrics

# Ensemble Model Building
def build_ensemble_models(X, y, fitted_models, X_train, X_test, y_train, y_test):
    """Build ensemble models from individual growth models"""
    # Create features from individual model predictions
    train_features = np.column_stack([
        model['function'](X_train, *model['parameters']) 
        for model in fitted_models.values()
    ])
    
    test_features = np.column_stack([
        model['function'](X_test, *model['parameters']) 
        for model in fitted_models.values()
    ])
    
    # Simple average ensemble
    simple_avg_train_pred = np.mean(train_features, axis=1)
    simple_avg_test_pred = np.mean(test_features, axis=1)
    
    # Weighted average ensemble (based on R² values)
    r2_values = np.array([metrics['test_r2'] for metrics in model_metrics.values()])
    # Adjust negative R² values
    r2_values = np.maximum(r2_values, 0)
    if np.sum(r2_values) > 0:
        weights = r2_values / np.sum(r2_values)
        weighted_avg_train_pred = np.sum(train_features * weights.reshape(1, -1), axis=1)
        weighted_avg_test_pred = np.sum(test_features * weights.reshape(1, -1), axis=1)
    else:
        weighted_avg_train_pred = simple_avg_train_pred
        weighted_avg_test_pred = simple_avg_test_pred
    
    # Machine learning ensembles
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(train_features, y_train)
    rf_train_pred = rf_model.predict(train_features)
    rf_test_pred = rf_model.predict(test_features)
    
    # Gradient Boosting
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(train_features, y_train)
    gb_train_pred = gb_model.predict(train_features)
    gb_test_pred = gb_model.predict(test_features)
    
    # Evaluate ensemble models
    ensemble_metrics = {
        'Simple Average': {
            'train_rmse': np.sqrt(mean_squared_error(y_train, simple_avg_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, simple_avg_test_pred)),
            'train_r2': r2_score(y_train, simple_avg_train_pred),
            'test_r2': r2_score(y_test, simple_avg_test_pred),
            'test_mae': mean_absolute_error(y_test, simple_avg_test_pred)
        },
        'Weighted Average': {
            'train_rmse': np.sqrt(mean_squared_error(y_train, weighted_avg_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, weighted_avg_test_pred)),
            'train_r2': r2_score(y_train, weighted_avg_train_pred),
            'test_r2': r2_score(y_test, weighted_avg_test_pred),
            'test_mae': mean_absolute_error(y_test, weighted_avg_test_pred)
        },
        'Random Forest': {
            'train_rmse': np.sqrt(mean_squared_error(y_train, rf_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, rf_test_pred)),
            'train_r2': r2_score(y_train, rf_train_pred),
            'test_r2': r2_score(y_test, rf_test_pred),
            'test_mae': mean_absolute_error(y_test, rf_test_pred)
        },
        'Gradient Boosting': {
            'train_rmse': np.sqrt(mean_squared_error(y_train, gb_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, gb_test_pred)),
            'train_r2': r2_score(y_train, gb_train_pred),
            'test_r2': r2_score(y_test, gb_test_pred),
            'test_mae': mean_absolute_error(y_test, gb_test_pred)
        }
    }
    
    # Print ensemble model metrics
    print("\nEnsemble Model Metrics:")
    for model_name, metrics in ensemble_metrics.items():
        print(f"\n{model_name} Ensemble:")
        print(f"  Train RMSE: {metrics['train_rmse']:.2f}")
        print(f"  Test RMSE: {metrics['test_rmse']:.2f}")
        print(f"  Train R²: {metrics['train_r2']:.4f}")
        print(f"  Test R²: {metrics['test_r2']:.4f}")
        print(f"  Test MAE: {metrics['test_mae']:.2f}")
    
    ensemble_models = {
        'Simple Average': {
            'train_pred': simple_avg_train_pred,
            'test_pred': simple_avg_test_pred
        },
        'Weighted Average': {
            'train_pred': weighted_avg_train_pred,
            'test_pred': weighted_avg_test_pred
        },
        'Random Forest': {
            'model': rf_model,
            'train_pred': rf_train_pred,
            'test_pred': rf_test_pred
        },
        'Gradient Boosting': {
            'model': gb_model,
            'train_pred': gb_train_pred,
            'test_pred': gb_test_pred
        }
    }
    
    return ensemble_models, ensemble_metrics

# Visualization Functions
def visualize_model_fits(df, X, y, X_train, X_test, y_train, y_test, fitted_models):
    """Visualize individual model fits"""
    plt.figure(figsize=(16, 10))
    plt.subplot(111)
    
    # Plot actual data
    plt.scatter(df['date'][X_train], y_train, color='blue', alpha=0.6, label='Training Data')
    plt.scatter(df['date'][X_test], y_test, color='red', alpha=0.6, label='Testing Data')
    
    # Predict for full date range and plot fitted models
    colors = ['green', 'purple', 'orange', 'brown']
    for i, (name, model) in enumerate(fitted_models.items()):
        y_pred = model['function'](X, *model['parameters'])
        plt.plot(df['date'], y_pred, color=colors[i % len(colors)], linewidth=2, label=f'{name} Model')
    
    plt.title('HIV Growth Models Comparison - Enugu State, Nigeria (2007-2023)', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Number of HIV Patients', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('hiv_individual_models_comparison.png')
    plt.close()

def visualize_ensemble_comparison(df, X, y, X_train, X_test, y_train, y_test, fitted_models, ensemble_models):
    """Visualize and compare ensemble models performance"""
    plt.figure(figsize=(16, 10))
    
    # Plot actual data
    plt.scatter(df['date'][X_train], y_train, color='blue', alpha=0.6, label='Training Data')
    plt.scatter(df['date'][X_test], y_test, color='red', alpha=0.6, label='Testing Data')
    
    # Plot best individual model
    best_model_name = max(model_metrics, key=lambda k: model_metrics[k]['test_r2'])
    best_model = fitted_models[best_model_name]
    y_pred_best = best_model['function'](X, *best_model['parameters'])
    plt.plot(df['date'], y_pred_best, color='green', linewidth=2, label=f'Best Individual ({best_model_name})')
    
    # Plot ensemble models
    colors = ['purple', 'orange', 'brown', 'magenta']
    ensemble_models_to_plot = ['Simple Average', 'Weighted Average', 'Random Forest', 'Gradient Boosting']
    
    # Create features for full range prediction
    full_features = np.column_stack([
        model['function'](X, *model['parameters']) 
        for model in fitted_models.values()
    ])
    
    for i, name in enumerate(ensemble_models_to_plot):
        if name in ['Simple Average', 'Weighted Average']:
            # For averaging ensembles, we need to recalculate
            if name == 'Simple Average':
                y_pred = np.mean(full_features, axis=1)
            else:  # Weighted Average
                r2_values = np.array([metrics['test_r2'] for metrics in model_metrics.values()])
                r2_values = np.maximum(r2_values, 0)
                if np.sum(r2_values) > 0:
                    weights = r2_values / np.sum(r2_values)
                    y_pred = np.sum(full_features * weights.reshape(1, -1), axis=1)
                else:
                    y_pred = np.mean(full_features, axis=1)
        else:  # Machine learning ensembles
            y_pred = ensemble_models[name]['model'].predict(full_features)
            
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
    plt.savefig('hiv_ensemble_models_comparison.png')
    plt.close()

def visualize_metrics_comparison(model_metrics, ensemble_metrics):
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
    plt.savefig('hiv_model_metrics_comparison.png')
    plt.close()

def forecast_future_trends(df, X, y, best_model, best_ensemble_model):
    """Forecast future HIV trends"""
    # Create future time points (next 5 years)
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=5*365, freq='D')
    
    # Create time index for future dates
    max_time_idx = df['time_idx'].max()
    future_time_idx = np.arange(max_time_idx + 1, max_time_idx + 1 + len(future_dates))
    
    # Predict using best individual model
    if isinstance(best_model, dict) and 'function' in best_model and 'parameters' in best_model:
        best_model_future = best_model['function'](future_time_idx, *best_model['parameters'])
    else:
        print("Error: Invalid best model format")
        return
    
    # Predict using best ensemble model (if it's an ML model)
    if isinstance(best_ensemble_model, dict) and 'model' in best_ensemble_model:
        # Generate features from individual models
        future_features = np.column_stack([
            model['function'](future_time_idx, *model['parameters']) 
            for model in fitted_models.values()
        ])
        best_ensemble_future = best_ensemble_model['model'].predict(future_features)
    else:
        print("Warning: Best ensemble model is not an ML model, skipping forecast")
        best_ensemble_future = None
    
    # Create future DataFrame
    future_df = pd.DataFrame({
        'date': future_dates,
        'time_idx': future_time_idx,
        'best_model_forecast': best_model_future
    })
    
    if best_ensemble_future is not None:
        future_df['best_ensemble_forecast'] = best_ensemble_future
    
    # Plot forecasts
    plt.figure(figsize=(16, 8))
    
    # Historical data
    plt.scatter(df['date'], y, color='blue', alpha=0.5, label='Historical Data')
    
    # Historical fitted values
    historical_fitted = best_model['function'](X, *best_model['parameters'])
    plt.plot(df['date'], historical_fitted, color='green', linestyle='-', linewidth=2, label='Best Model Fit')
    
    # Future forecasts
    plt.plot(future_df['date'], future_df['best_model_forecast'], color='green', linestyle='--', linewidth=2, label='Best Model Forecast')
    
    if best_ensemble_future is not None:
        plt.plot(future_df['date'], future_df['best_ensemble_forecast'], color='purple', linestyle='--', linewidth=2, label='Best Ensemble Forecast')
    
    plt.title('HIV Trend Forecast (2024-2028) - Enugu State, Nigeria', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Number of HIV Patients', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))
    plt.xticks(rotation=45)
    
    # Add forecast uncertainty band
    if best_ensemble_future is not None:
        uncertainty = np.abs(future_df['best_model_forecast'] - future_df['best_ensemble_forecast']) * 1.5
        plt.fill_between(future_df['date'], 
                         future_df['best_model_forecast'] - uncertainty,
                         future_df['best_model_forecast'] + uncertainty,
                         color='gray', alpha=0.3, label='Forecast Uncertainty')
    
    plt.tight_layout()
    plt.savefig('hiv_forecast_2024_2028.png')
    plt.close()
    
    return future_df

# Main function
def main():
    # Load and explore data
    file_path = 'data.xlsx'
    df = load_data(file_path)
    
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Prepare data for modeling
    X, y, X_train, X_test, y_train, y_test = prepare_data_for_modeling(df)
    
    # Fit individual growth models
    global fitted_models, model_metrics
    fitted_models, model_metrics = fit_growth_models(X_train, y_train, X_test, y_test)
    
    # Visualize individual model fits
    visualize_model_fits(df, X, y, X_train, X_test, y_train, y_test, fitted_models)
    
    # Build ensemble models
    ensemble_models, ensemble_metrics = build_ensemble_models(X, y, fitted_models, X_train, X_test, y_train, y_test)
    
    # Visualize ensemble comparisons
    visualize_ensemble_comparison(df, X, y, X_train, X_test, y_train, y_test, fitted_models, ensemble_models)
    
    # Compare model metrics
    visualize_metrics_comparison(model_metrics, ensemble_metrics)
    
    # Identify best models
    best_individual_model_name = max(model_metrics, key=lambda k: model_metrics[k]['test_r2'])
    best_ensemble_model_name = max(ensemble_metrics, key=lambda k: ensemble_metrics[k]['test_r2'])
    
    print(f"\nBest Individual Model: {best_individual_model_name}")
    print(f"Best Ensemble Model: {best_ensemble_model_name}")
    
    # Forecast future trends
    future_df = forecast_future_trends(
        df, X, y, 
        fitted_models[best_individual_model_name],
        ensemble_models[best_ensemble_model_name]
    )
    
    print("\nAnalysis complete. All visualizations saved to disk.")

if __name__ == "__main__":
    main()
