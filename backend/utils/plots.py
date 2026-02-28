# Import required libraries
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

    # Add legend, rotate ticks, and show the plot
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

    # Forecasted sell-out line
    ax.plot(days, y_pred_prod, label='Previsione', color=palette[0], linewidth=2, linestyle='--')

    # Shade area representing forecast + safety stock (as a band of uncertainty)
    ax.fill_between(days, y_pred_prod, y_pred_prod + safety_stock, 
                    color=palette[1], alpha=0.2, label='Intervallo Safety Stock')

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
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=17)
    plt.xticks(rotation=45, ha='right')

    # Add a light grid
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.close()
    
    return fig


def visualize_forecast_fanchart(
    product: str,
    y_true: ArrayLike,
    p50: ArrayLike,
    p10: ArrayLike | None = None,
    p90: ArrayLike | None = None,
    save_path: str | None = None,
):
    y_true = np.asarray(y_true)
    p50 = np.asarray(p50)
    x = np.arange(1, len(p50) + 1)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(x, y_true, label="Actual", color="#2F6690", linewidth=2)
    ax.plot(x, p50, label="P50", color="#F26419", linewidth=2, linestyle="--")

    if p10 is not None and p90 is not None:
        p10 = np.asarray(p10)
        p90 = np.asarray(p90)
        lower = np.minimum(p10, p90)
        upper = np.maximum(p10, p90)

        ax.fill_between(x, lower, upper, color="#F6AE2D", alpha=0.25, label="P10-P90 band")

        outside = (y_true < lower) | (y_true > upper)
        if np.any(outside):
            ax.scatter(x[outside], y_true[outside], color="red", s=35, label="Outside P10-P90")

    ax.set_title(f"Probabilistic Forecast (Fan Chart) - {product}", fontsize=16, fontweight="bold")
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel("Demand", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    plt.close()
    return fig


def visualize_inventory_scenarios(
    product: str,
    stock_low: ArrayLike,
    stock_base: ArrayLike,
    stock_high: ArrayLike,
    safety_stock_line: ArrayLike | None = None,
):
    stock_low = np.asarray(stock_low, dtype=float)
    stock_base = np.asarray(stock_base, dtype=float)
    stock_high = np.asarray(stock_high, dtype=float)
    x = np.arange(1, len(stock_base) + 1)

    band_low = np.minimum(stock_low, stock_high)
    band_high = np.maximum(stock_low, stock_high)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.fill_between(x, band_low, band_high, color="#A9D6E5", alpha=0.35, label="Inventory band (LOW-HIGH)")
    ax.plot(x, stock_low, color="#2A9D8F", linewidth=2, linestyle="--", label="LOW demand stock")
    ax.plot(x, stock_base, color="#1D3557", linewidth=2, label="BASE demand stock")
    ax.plot(x, stock_high, color="#E76F51", linewidth=2, linestyle="-.", label="HIGH demand stock")

    if safety_stock_line is not None:
        ss = np.asarray(safety_stock_line, dtype=float)
        if ss.size == 1:
            ss = np.full_like(stock_base, float(ss[0]))
        ax.plot(x, ss, color="#6D6875", linewidth=1.8, linestyle=":", label="Safety stock")

    high_stockout_idx = np.where(stock_high <= 0)[0]
    if high_stockout_idx.size > 0:
        i0 = int(high_stockout_idx[0])
        ax.scatter([x[i0]], [stock_high[i0]], color="red", s=50, zorder=5)
        ax.annotate(
            f"First HIGH stockout: week {x[i0]}",
            xy=(x[i0], stock_high[i0]),
            xytext=(10, 10),
            textcoords="offset points",
            color="red",
            fontsize=10,
        )

    ax.set_title(f"Inventory Scenario Band - {product}", fontsize=16, fontweight="bold")
    ax.set_xlabel("Week", fontsize=14)
    ax.set_ylabel("Stock", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.close()
    return fig


def plot_coverage(
    product: str,
    coverage_global: float,
    expected: float = 0.80,
    coverage_by_h: ArrayLike | None = None,
):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(expected, color="#B22222", linestyle="--", linewidth=2, label=f"Expected ({expected:.2f})")

    if coverage_by_h is not None:
        cov = np.asarray(coverage_by_h, dtype=float)
        h = np.arange(1, len(cov) + 1)
        ax.plot(h, cov, color="#1F77B4", linewidth=2, marker="o", label="Coverage by horizon")
        ax.set_xlabel("Horizon (h)")
    else:
        cov = np.asarray([coverage_global], dtype=float)
        h = np.asarray([1])
        ax.bar(h, cov, color="#1F77B4", alpha=0.85, label="Coverage")
        ax.set_xticks([1])
        ax.set_xticklabels(["Global"])
        ax.set_xlabel("")

    if not np.isnan(coverage_global):
        ax.text(
            0.02,
            0.95,
            f"Global coverage: {coverage_global:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Coverage")
    ax.set_title(f"P10-P90 Coverage - {product}")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.close()
    return fig


def plot_error_vs_horizon(product: str, abs_error_by_h: ArrayLike):
    err = np.asarray(abs_error_by_h, dtype=float)
    h = np.arange(1, len(err) + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(h, err, color="#2E8B57", marker="o", linewidth=2)
    ax.set_xlabel("Horizon (h)")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title(f"Absolute Error vs Horizon - {product}")
    ax.grid(True, linestyle="--", alpha=0.5)
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

    # Export forecast and actual values to CSV
    df = pd.DataFrame({
        "Sellout Effettivo": np.concatenate([y[:split_idx], [np.nan] * (len(y_pred_adjusted[split_idx:]) - len(y[:split_idx]))]),
        "Previsione Sellout": np.concatenate([y_pred_adjusted[split_idx:]])
    })
    df.to_csv(r"C:\Users\fraca\Desktop\sales\Sell_Out_Sell_In_Forecasting-main\output\forecast_data.csv", index=False)

    plt.tight_layout()
    return fig

# -----------------------------------------------------------------------
# 4. Plot feature importance from a predictive model
# -----------------------------------------------------------------------
def plot_feat_importance(importance: ArrayLike,
                         feature_names: ArrayLike,
                         min_importance: float = 0.01,
                         n_features: int = 10,
                         save_path: str | None = None) -> None:
    df_feature_importance = pd.DataFrame({
        'column': feature_names,
        'importance': importance
    })
    df_feature_importance = df_feature_importance[
        df_feature_importance['importance'] >= min_importance
    ].nlargest(n_features, 'importance').sort_values('importance', ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.barh(df_feature_importance['column'], df_feature_importance['importance'])
    ax.tick_params(axis='both', labelsize=10)
    ax.set_title('Model Feature Importance', size=20)
    ax.set_xlabel('Importance', size=15)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

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
