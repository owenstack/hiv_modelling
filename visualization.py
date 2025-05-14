import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from utilities import generate_bootstrap_predictions

# Set styling for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')

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
def plot_basic_timeseries(df):
    """Create basic time series plot of the data"""
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['cumulative'], marker='o', linestyle='-', alpha=0.7)
    plt.title('Cumulative HIV Cases Over Time in Enugu State, Nigeria (2007-2023)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Number of HIV Patients')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

@plot_manager
def visualize_model_fits(df, X, y, X_train, X_test, y_train, y_test, fitted_models, filename='hiv_individual_models_comparison.png'):
    """Visualize individual model fits"""
    fig = plt.figure(figsize=(16, 10))
    
    plt.scatter(df['date'].iloc[X_train], y_train, color='blue', alpha=0.6, label='Training Data')
    plt.scatter(df['date'].iloc[X_test], y_test, color='red', alpha=0.6, label='Testing Data')
    
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
def visualize_ensemble_comparison(df, X, y, cv_splits, fitted_models, ensemble_models, best_model_name):
    """Visualize and compare ensemble models performance"""
    fig = plt.figure(figsize=(16, 10))
    
    train_idx, test_idx = cv_splits[-1]
    
    plt.scatter(df['date'].iloc[train_idx], y[train_idx], color='blue', alpha=0.6, label='Training Data')
    plt.scatter(df['date'].iloc[test_idx], y[test_idx], color='red', alpha=0.6, label='Testing Data')
    
    best_model = fitted_models[best_model_name]
    y_pred_best = best_model['function'](X, *best_model['parameters'])
    plt.plot(df['date'], y_pred_best, color='green', linewidth=2, label=f'Best Individual ({best_model_name})')
    
    # Create features for ensemble predictions
    base_features = np.column_stack([
        model['function'](X, *model['parameters']) 
        for model in fitted_models.values()
    ])
    
    colors = ['purple', 'orange', 'brown', 'magenta']
    ensemble_models_to_plot = ['Simple Average', 'Weighted Average', 'Random Forest', 'Gradient Boosting']
    
    for i, name in enumerate(ensemble_models_to_plot):
        if name in ['Simple Average', 'Weighted Average']:
            y_pred = ensemble_models[name]['predict'](base_features)
        else:
            features = ensemble_models[name]['create_features'](X)
            scaled_features = ensemble_models[name]['scaler'].transform(features)
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
    all_metrics = {**model_metrics, **ensemble_metrics}
    
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
    
    fig = plt.figure(figsize=(18, 15))
    gs = GridSpec(2, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    sns.barplot(x='Model', y='RMSE', hue='Type', data=metrics_df, ax=ax1)
    ax1.set_title('Root Mean Squared Error (RMSE) Comparison', fontsize=14)
    ax1.set_xlabel('')
    ax1.set_ylabel('RMSE (lower is better)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    ax2 = fig.add_subplot(gs[0, 1])
    sns.barplot(x='Model', y='R²', hue='Type', data=metrics_df, ax=ax2)
    ax2.set_title('R² Score Comparison', fontsize=14)
    ax2.set_xlabel('')
    ax2.set_ylabel('R² (higher is better)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    ax3 = fig.add_subplot(gs[1, 0])
    sns.barplot(x='Model', y='MAE', hue='Type', data=metrics_df, ax=ax3)
    ax3.set_title('Mean Absolute Error (MAE) Comparison', fontsize=14)
    ax3.set_xlabel('')
    ax3.set_ylabel('MAE (lower is better)', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    
    ax4 = fig.add_subplot(gs[1, 1])
    
    metrics_df['RMSE_norm'] = (metrics_df['RMSE'] - metrics_df['RMSE'].min()) / (metrics_df['RMSE'].max() - metrics_df['RMSE'].min() + 1e-10)
    metrics_df['R²_norm'] = 1 - (metrics_df['R²'] - metrics_df['R²'].min()) / (metrics_df['R²'].max() - metrics_df['R²'].min() + 1e-10)
    metrics_df['MAE_norm'] = (metrics_df['MAE'] - metrics_df['MAE'].min()) / (metrics_df['MAE'].max() - metrics_df['MAE'].min() + 1e-10)
    metrics_df['Overall_Score'] = (1 - metrics_df['RMSE_norm'] + metrics_df['R²_norm'] + (1 - metrics_df['MAE_norm'])) / 3
    
    metrics_df = metrics_df.sort_values('Overall_Score', ascending=False)
    
    sns.barplot(x='Model', y='Overall_Score', hue='Type', data=metrics_df, ax=ax4)
    ax4.set_title('Overall Model Performance Score', fontsize=14)
    ax4.set_xlabel('')
    ax4.set_ylabel('Performance Score (higher is better)', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

@plot_manager
def create_validation_plot(df, X, y, best_model, best_ensemble_model, ensemble_name, fitted_models, best_model_name, filename='hiv_model_validation.png'):
    """Create validation plot comparing actual vs predicted values for historical data"""
    historical_best_model = best_model['function'](X, *best_model['parameters'])
    
    historical_features = np.column_stack([
        model['function'](X, *model['parameters']) 
        for model in fitted_models.values()
    ])
    
    if ensemble_name in ['Random Forest', 'Gradient Boosting']:
        scaled_features = best_ensemble_model['scaler'].transform(historical_features)
        historical_ensemble = best_ensemble_model['model'].predict(scaled_features)
    else:
        historical_ensemble = best_ensemble_model['predict'](historical_features)
    
    fig = plt.figure(figsize=(16, 8))
    
    plt.plot(df['date'], y, color='blue', linewidth=2, label='Actual Cumulative HIV Cases')
    
    plt.plot(df['date'], historical_best_model, color='green', linestyle='--', 
             linewidth=2, label=f'Best Individual Model ({best_model_name})')
    plt.plot(df['date'], historical_ensemble, color='purple', linestyle='--', 
             linewidth=2, label=f'Best Ensemble Model ({ensemble_name})')
    
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
    
    plt.figtext(0.15, 0.15, f"Best Individual Model RMSE: {individual_rmse:.2f}\nMAE: {individual_mae:.2f}",
                bbox=dict(facecolor='white', alpha=0.8))
    plt.figtext(0.15, 0.05, f"Best Ensemble Model RMSE: {ensemble_rmse:.2f}\nMAE: {ensemble_mae:.2f}",
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

@plot_manager
def forecast_future_trends(df, X, y, best_model, best_ensemble_model, ensemble_name, fitted_models, best_model_name, generate_bootstrap_predictions, filename='hiv_forecast_2024_2028.png'):
    """Forecast future HIV trends with enhanced uncertainty quantification"""
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=5*365, freq='D')
    
    max_time_idx = df['time_idx'].max()
    future_time_idx = np.arange(max_time_idx + 1, max_time_idx + 1 + len(future_dates))
    
    _, lower_bound, upper_bound = generate_bootstrap_predictions(
        best_model['function'], future_time_idx, best_model['parameters'], n_samples=1000
    )
    
    best_model_future = best_model['function'](future_time_idx, *best_model['parameters'])
    
    future_features = np.column_stack([
        model['function'](future_time_idx, *model['parameters']) 
        for model in fitted_models.values()
    ])
    
    if ensemble_name in ['Random Forest', 'Gradient Boosting']:
        scaled_features = best_ensemble_model['scaler'].transform(future_features)
        best_ensemble_future = best_ensemble_model['model'].predict(scaled_features)
    else:
        best_ensemble_future = best_ensemble_model['predict'](future_features)
    
    future_df = pd.DataFrame({
        'date': future_dates,
        'time_idx': future_time_idx,
        'best_model_forecast': best_model_future,
        'best_ensemble_forecast': best_ensemble_future,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    })
    
    fig = plt.figure(figsize=(16, 8))
    
    plt.scatter(df['date'], y, color='blue', alpha=0.5, label='Historical Data')
    
    historical_fitted = best_model['function'](X, *best_model['parameters'])
    plt.plot(df['date'], historical_fitted, color='green', linestyle='-', linewidth=2, 
             label=f'Best Individual Model Fit ({best_model_name})')
    
    plt.plot(future_df['date'], future_df['best_model_forecast'], color='green', linestyle='--', 
             linewidth=2, label=f'Best Individual Model Forecast ({best_model_name})')
    plt.plot(future_df['date'], future_df['best_ensemble_forecast'], color='purple', linestyle='--', 
             linewidth=2, label=f'Best Ensemble Forecast ({ensemble_name})')
    
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
    
    future_df.to_csv('data/forecast_results_2024_2028.csv', index=False)
    return fig