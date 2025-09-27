# =============================================
# 📦 Inventory Forecasting Utility Functions
# =============================================

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error

# ---------------------------------------------
# 🎯 Demand Planning Accuracy (DPA)
# ---------------------------------------------
def DPA(y_true, y_pred):
    """
    Measures how much of the true demand is correctly predicted.
    DPA = 100% means perfect prediction.
    """
    dpa = 100 - (((np.sum(np.abs(y_pred - y_true))) / np.sum(y_true)) * 100)
    return dpa

# ---------------------------------------------
# ⚖️ Forecast Bias
# ---------------------------------------------
def BIAS(y_true, y_pred):
    """
    Measures systematic over- or under-forecasting.
    Positive bias → overestimation, negative → underestimation.
    """
    bias = (((np.sum(y_pred - y_true)) / np.sum(y_true)) * 100)
    return bias

# ---------------------------------------------
# 📉 Root Mean Squared Error
# ---------------------------------------------
def RMSE(y_true, y_pred):
    """
    Measures average magnitude of forecast errors.
    Penalizes larger errors more heavily.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

# ---------------------------------------------
# 📊 Mean Absolute Percentage Error
# ---------------------------------------------
def MAPE(y_true, y_pred):
    """
    Expresses forecast error as a percentage of actual demand.
    Robust and interpretable.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0  # Avoid division by zero
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + 1e-10))) * 100

# ---------------------------------------------
# 🚫 Lost Sales Calculation (Simple)
# ---------------------------------------------
def calculate_lost_sales(y_pred, sellout):
    """
    Estimates the demand that couldn’t be met because of under-forecasting.
    Assumes lost sales only occur when forecast is lower than actual.
    """
    lost_sales = y_pred - sellout
    return np.maximum(lost_sales, 0)  # Clamp negatives to zero

# ---------------------------------------------
# 🚫 Lost Sales with Inventory Tracking
# ---------------------------------------------
def calculate_lost_sales_with_stock(y_pred, sellout, initial_stock, sell_in):
    """
    Models lost sales by simulating a rolling inventory system:
    - Adds sell-in each week
    - Removes actual sell-out
    - Calculates if stock was insufficient
    """
    stock = initial_stock
    lost_sales = np.zeros(len(y_pred))

    for i in range(len(y_pred)):
        potential_demand = sellout[i]         # Real demand that needs to be fulfilled
        available = stock + y_pred[i]         # Forecasted supply available this week
        
        if available < potential_demand:
            lost_sales[i] = potential_demand - available
            stock = 0                         # All stock used up
        else:
            stock += available - sellout[i]   # Remaining stock carries over
    
    if isinstance(sell_in, pd.Series):
        lost_sales = pd.Series(lost_sales, index=sell_in.index)
    
    return lost_sales


def WAPE(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    denom = np.sum(np.abs(y)) + 1e-8
    return 100.0 * np.sum(np.abs(y - yhat)) / denom

def SMAPE(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    denom = (np.abs(y) + np.abs(yhat)) / 2.0 + 1e-8
    return 100.0 * np.mean(np.abs(y - yhat) / denom)

def MASE(y, yhat, y_insample=None, m=1):
    # se non passi y_insample, approssima con y stessa (non ideale ma ok per backtest semplice)
    if y_insample is None or len(y_insample) < m + 1:
        y_insample = y
    d = np.mean(np.abs(y_insample[m:] - y_insample[:-m])) + 1e-8
    return np.mean(np.abs(np.asarray(y) - np.asarray(yhat))) / d

# ---------------------------------------------
# 📈 Service Level
# ---------------------------------------------
def calculate_service_level(sellout, lost_sales):
    """
    Service level represents the % of demand successfully served from inventory.
    A key metric in supply chain reliability.
    """
    return (sellout / (sellout + lost_sales)) * 100

# ---------------------------------------------
# 📦 Weekly Inventory & Stock Cover
# ---------------------------------------------
def calculate_weekly_stock_and_cover(sell_in, sell_out):
    """
    Simulates weekly inventory (stock) and calculates how long it will last (cover).
    
    Stock: Amount of product available at the end of each week.
    Stock cover: How many future weeks can be served with current stock.
    """
    sell_in = np.array(sell_in, dtype=float)
    sell_out = np.array(sell_out, dtype=float)

    n = len(sell_in)
    stock = np.zeros(n, dtype=float)
    cumulative_stock = 0.0  # Tracks ongoing inventory level

    for i in range(n):
        diff = sell_in[i] - sell_out[i]
        if diff < 0:
            cumulative_stock = 0.0  # Stock-out: inventory is depleted
        else:
            cumulative_stock += diff
        stock[i] = max(0.0, cumulative_stock)

    # Stock cover: how many weeks of future demand we can meet with today’s stock
    stock_cover = np.zeros(n, dtype=float)

    for i in range(n):
        starting_stock = stock[i]
        if starting_stock <= 0:
            stock_cover[i] = 0.0
            continue
        
        cumulato = 0.0
        cover = 0.0

        for j in range(i, n):
            venduto = sell_out[j]
            
            if venduto <= 0:
                continue  # Skip zero or negative demand (no consumption)

            if cumulato + venduto >= starting_stock:
                # Only part of this week’s demand is covered → fractional week
                residuo = starting_stock - cumulato
                frazione_settimana = residuo / venduto
                cover = (j - i) + frazione_settimana
                break

            cumulato += venduto  # Accumulate until stock is exhausted

        if cover == 0.0 and cumulato < starting_stock:
            # Stock lasts beyond visible demand horizon
            cover = n - i

        stock_cover[i] = cover

    return stock, stock_cover

# ---------------------------------------------
# 🛡️ Safety Stock Calculation
# ---------------------------------------------
def calculate_safety_stock(sellout_history, sell_in_history, service_level, forecast_period):
    """
    Computes safety stock based on variability in historical sell-out and service level.
    Formula:
        SS = z * √(forecast_period * σ² + demand buffer)
    """
    sellout_std = np.std(sellout_history)  # Demand variability (volatility)
    z_score = stats.norm.ppf(service_level)  # Service level to Z-score

    # Additive model: standard deviation effect + slight demand buffer
    safety_stock = z_score * np.sqrt(
        forecast_period * (sellout_std ** 2) + np.mean(sellout_history) * 1
    )

    return safety_stock
