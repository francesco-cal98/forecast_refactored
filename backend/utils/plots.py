# Import required libraries
# Use the non-interactive Agg backend so backtest runs never open GUI windows.
# This prevents Tk event-loop crashes when multiple plots are created in a loop.
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
import seaborn as sns

# -----------------------------------
# 1. Plot forecast vs actual values
# -----------------------------------
def plot_forecast_vs_actual(y: ArrayLike, 
                            y_pred: ArrayLike,
                            x: ArrayLike | None = None, 
                            additional_graphics: dict[str, str] | None = None) -> None:
    # Convert inputs to numpy arrays
    y = np.asarray(y)
    y_pred = np.asarray(y_pred)
    x = np.arange(len(y)) if x is None else np.asarray(x)

    # Plot actual values
    plt.plot(x, y, label='Actual', alpha=0.7)
    # Plot forecast values
    plt.plot(x, y_pred, label='Forecast', alpha=0.7)

    # If title not provided, default to 'None'
    if additional_graphics is None:
        additional_graphics = {'title': 'None'}

    # Add plot title and axis labels
    plt.title(f'Actual vs Forecast - {additional_graphics["title"]} set')
    plt.xlabel('Date')
    plt.ylabel('Value')

    # Add legend, rotate ticks, and close (caller is responsible for saving)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.close()

# -----------------------------------------------------------------------
# 2. Visualize forecast + safety stock + stock level + sell-in strategy
# -----------------------------------------------------------------------
def visualize_forecast_and_safety_stock(product: str, 
                                        y_pred_prod: ArrayLike,
                                        y: ArrayLike, 
                                        weekly_stock: ArrayLike, 
                                        safety_stock: ArrayLike,
                                        required_sell_in: ArrayLike,
                                        stock_cover: ArrayLike | None = None):
    # Set a clean seaborn style
    sns.set_style("whitegrid")
    
    # Use a colorblind-friendly palette
    palette = sns.color_palette("colorblind", 6)
    fig, ax = plt.subplots(figsize=(15, 8))  # Create a large figure

    # X-axis for plotting
    days = range(1, len(y_pred_prod) + 1)

    y_pred_prod = np.asarray(y_pred_prod, dtype=float)
    y = np.asarray(y, dtype=float)
    weekly_stock = np.asarray(weekly_stock, dtype=float)
    required_sell_in = np.asarray(required_sell_in, dtype=float)
    safety_stock = np.asarray(safety_stock, dtype=float)
    if safety_stock.ndim == 0:
        safety_stock = np.full_like(y_pred_prod, float(safety_stock))
    elif safety_stock.size != y_pred_prod.size:
        safety_stock = np.full_like(y_pred_prod, float(np.nanmean(safety_stock)))

    # Forecasted sell-out line
    ax.plot(days, y_pred_prod, label='Previsione', color=palette[0], linewidth=2, linestyle='--')

    # Safety stock floor (flat/step line in level units).
    ax.plot(days, safety_stock, label='Safety Stock', color=palette[1], linewidth=2)

    # Required sell-in line (predicted stock to order)
    ax.plot(days, required_sell_in, label='Stima degli ordini', color=palette[2], linewidth=2, linestyle='--')


    # Actual sell-out values
    ax.plot(days, y, label='Sellout Effettivo', color=palette[3], linewidth=2)

    # Weekly stock levels after each week
    ax.plot(days, weekly_stock, label='Stock Settimanale', color=palette[4], linewidth=2, linestyle='--')


    # Titles and labels
    ax.set_title(f'Previsione, Stock e Safety Stock per {product}', fontsize=18, fontweight='bold')
    ax.set_xlabel('Data', fontsize=17)
    ax.set_ylabel('Quantità', fontsize=17)
    
    # Ticks and legend styling
    ax.tick_params(axis='both', which='major', labelsize=17)
    handles, labels = ax.get_legend_handles_labels()
    if stock_cover is not None:
        cover_arr = np.asarray(stock_cover, dtype=float)
        if cover_arr.size == len(days):
            ax2 = ax.twinx()
            ax2.plot(days, cover_arr, label='Stock Cover (weeks)', color=palette[5], linewidth=1.5, alpha=0.9)
            ax2.set_ylabel('Stock Cover (weeks)', fontsize=14)
            ax2.tick_params(axis='y', labelsize=12)
            h2, l2 = ax2.get_legend_handles_labels()
            handles += h2
            labels += l2
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=17)
    plt.xticks(rotation=45, ha='right')

    # Add a light grid
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.close()
    
    return fig

# -----------------------------------------------------------------------
# 3. Second visualization: historical vs forecasted + export to CSV
# -----------------------------------------------------------------------
def visualize_forecast_and_safety_stock2(product: str, 
                                         y_pred_prod: ArrayLike,
                                         y: ArrayLike, 
                                         weekly_stock: ArrayLike, 
                                         required_sell_in: ArrayLike,
                                         sell_in: ArrayLike):
    sns.set_style("whitegrid")
    palette = sns.color_palette("colorblind", 5)
    fig, ax = plt.subplots(figsize=(15, 8))

    # Time axis
    days = range(1, len(y_pred_prod) + 1)
    split_idx = len(y) // 2  # Approximate transition point

    # Plot actual sell-out data
    ax.plot(days[:split_idx], y[:split_idx], label='Sellout Effettivo', color=palette[0], linewidth=2)

    # Prepare forecasted values to ensure smooth transition from actual
    y_pred_adjusted = np.insert(y_pred_prod, 0, y[split_idx])
    ax.plot(days[split_idx-1:], y_pred_adjusted[split_idx:], 
            label='Previsione Sellout', color=palette[0], linewidth=2, linestyle='--')

    # Set title, labels, and style
    ax.set_title(f'Previsione, Stock e Safety Stock per {product}', fontsize=18, fontweight='bold')
    ax.set_xlabel('Data', fontsize=17)
    ax.set_ylabel('Quantità', fontsize=17)
    ax.tick_params(axis='both', which='major', labelsize=17)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=17)
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    return fig

# -----------------------------------------------------------------------
# 4. Plot feature importance from a predictive model
# -----------------------------------------------------------------------
def plot_feat_importance(importance: ArrayLike, 
                         feature_names: ArrayLike, 
                         min_importance: float = 0.01,
                         n_features: int = 10):
    # Prepare a DataFrame for plotting
    df_feature_importance = pd.DataFrame({
        'column': feature_names,
        'importance': importance
    })

    # Filter and sort top features by importance
    df_feature_importance = df_feature_importance[
        df_feature_importance['importance'] >= min_importance
    ].nlargest(n_features, 'importance').sort_values('importance', ascending=True).reset_index(drop=True)

    # Create horizontal bar chart and return the figure so the caller can save it
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.barh(df_feature_importance['column'], df_feature_importance['importance'])
    ax.tick_params(axis='both', labelsize=10)
    ax.set_title('Model Feature Importance', size=20)
    ax.set_xlabel('Importance', size=15)
    plt.tight_layout()
    return fig

# -----------------------------------------------------------------------
# 5. Scenario analysis: central, optimistic, pessimistic forecasts
# -----------------------------------------------------------------------
def plot_scenarios(product: str,
                   y_pred_prod: np.ndarray,
                   y: np.ndarray,
                   y_pred_prod_positive: np.ndarray,
                   y_pred_prod_negative: np.ndarray,
                   mape: float):
    # Time axis
    days = np.arange(1, len(y_pred_prod) + 1)

    # Create figure
    plt.figure(figsize=(10, 6))

    # Central forecast (neutral scenario)
    plt.plot(days, y_pred_prod, label="Neutral Scenario (Prediction)", color="#4A90E2")

    # Confidence interval shading between upper and lower bounds
    plt.fill_between(days, 
                     y_pred_prod_negative, 
                     y_pred_prod_positive, 
                     color="#A6C8FF", alpha=0.4, label="Confidence Interval")

    # Upper bound (positive scenario)
    plt.plot(days, y_pred_prod_positive, label="Upper Bound (Positive Scenario)", color="#6DBE45", linestyle='-', linewidth=2)

    # Lower bound (negative scenario)
    plt.plot(days, y_pred_prod_negative, label="Lower Bound (Negative Scenario)", color="#FF6F61", linestyle='-', linewidth=2)

    # Actual historical data
    plt.plot(days, y, color="#3B9A9C", label="Actual Values", alpha=0.6)

    # Add title and labels
    plt.title(f"Time Series Forecast with Confidence Intervals (MAPE: {mape:.2f}%)")
    plt.xlabel("Time")
    plt.ylabel("Predicted Value")
    plt.legend()
    plt.close()
