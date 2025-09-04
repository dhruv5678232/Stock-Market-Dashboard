import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# ======================================
# Utility Functions
# ======================================
def evaluate_regression(y_true, y_pred, model_name, feature=None):
    """Return regression metrics as dict"""
    return {
        "Model": model_name,
        "Feature": feature,
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False)
    }

def run_arima(series, test_size=0.2, forecast_horizon=10):
    """Fit ARIMA with simple (p,d,q) search"""
    series = series.dropna()
    n_test = int(len(series) * test_size)
    train, test = series[:-n_test], series[-n_test:]
    best_aic, best_order, best_model = np.inf, None, None

    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    model = ARIMA(train, order=(p, d, q))
                    fit = model.fit()
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_order = (p, d, q)
                        best_model = fit
                except:
                    continue

    if best_model is None:
        return None

    pred_test = best_model.predict(start=len(train), end=len(series)-1, typ="levels")
    forecast = best_model.forecast(steps=forecast_horizon)

    metrics = {
        "Model": f"ARIMA{best_order}",
        "MAE": mean_absolute_error(test, pred_test),
        "MSE": mean_squared_error(test, pred_test),
        "RMSE": mean_squared_error(test, pred_test, squared=False),
        "AIC": best_aic
    }

    return metrics, pred_test, test, forecast, best_order


# ======================================
# Streamlit App
# ======================================
st.title("📈 Stock Market Prediction Dashboard")

# 🔹 Load dataset (fixed path)
file_path = r"C:\Users\Dhruv Patel\OneDrive\Desktop\proj\synthetic_stock_data.csv"
df = pd.read_csv(file_path)
st.write("### Dataset Preview", df.head())

# Parse date column if available
for col in df.columns:
    if "date" in col.lower():
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df = df.set_index(col).sort_index()
        break

# Select numeric target column
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns available for prediction.")
    st.stop()

target = st.selectbox("🎯 Select target variable", numeric_cols)
X = df.drop(columns=[target])
y = df[target]

# Encode categoricals
X = pd.get_dummies(X, drop_first=True)

metrics_list = []

# ======================================
# SLR (best single feature)
# ======================================
for feature in X.columns:
    Xf = X[[feature]]
    X_train, X_test, y_train, y_test = train_test_split(Xf, y, test_size=0.2, random_state=42)
    lr = LinearRegression().fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    metrics_list.append(evaluate_regression(y_test, y_pred, "SLR", feature))

# Keep best SLR
best_slr = min([m for m in metrics_list if m["Model"]=="SLR"], key=lambda x: x["RMSE"])
st.subheader("🔹 Best Simple Linear Regression")
st.write(best_slr)

# ======================================
# MLR
# ======================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
mlr = LinearRegression().fit(X_train, y_train)
y_pred_mlr = mlr.predict(X_test)
mlr_metrics = evaluate_regression(y_test, y_pred_mlr, "MLR")
metrics_list.append(mlr_metrics)

# Plot Actual vs Predicted
fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(y_test, y_pred_mlr, alpha=0.6)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("MLR — Actual vs Predicted")
st.pyplot(fig)

# Residuals plot
residuals = y_test - y_pred_mlr
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(y_pred_mlr, residuals, alpha=0.6)
ax.axhline(0, color="red", linestyle="--")
ax.set_xlabel("Predicted")
ax.set_ylabel("Residual")
ax.set_title("MLR Residuals")
st.pyplot(fig)

# ======================================
# ARIMA
# ======================================
st.subheader("⏱️ ARIMA Forecasting")
forecast_horizon = st.slider("Forecast horizon (days)", 5, 30, 10)

arima_outputs = run_arima(y, test_size=0.2, forecast_horizon=int(forecast_horizon))
if arima_outputs is None:
    st.warning("ARIMA could not be fit (series too short or unsuitable).")
else:
    arima_metrics, pred_test, y_test_arima, future_forecast, best_order = arima_outputs

    # Plot test vs forecast
    fig, ax = plt.subplots(figsize=(8,4))
    if isinstance(y.index, pd.DatetimeIndex):
        test_index = y.iloc[-len(y_test_arima):].index
        ax.plot(test_index, y_test_arima, label="Actual (test)")
        ax.plot(test_index, pred_test, label=f"Predicted ARIMA{best_order}")
    else:
        ax.plot(range(len(y_test_arima)), y_test_arima.values, label="Actual (test)")
        ax.plot(range(len(pred_test)), pred_test.values, label=f"Predicted ARIMA{best_order}")
    ax.set_title("ARIMA — Test vs Predicted")
    ax.legend()
    st.pyplot(fig)

    # Plot future forecast
    fig, ax = plt.subplots(figsize=(8,4))
    if isinstance(y.index, pd.DatetimeIndex):
        ax.plot(y.index, y.values, label="History")
        future_index = pd.date_range(start=y.index[-1] + pd.Timedelta(days=1), periods=len(future_forecast), freq="D")
        ax.plot(future_index, future_forecast.values, label="Future Forecast")
    else:
        ax.plot(range(len(y)), y.values, label="History")
        ax.plot(range(len(y), len(y)+len(future_forecast)), future_forecast.values, label="Future Forecast")
    ax.set_title("ARIMA — Future Forecast")
    ax.legend()
    st.pyplot(fig)

# ======================================
# Comparison Table
# ======================================
st.subheader("📋 Model Comparison")
comp_rows = []
for m in metrics_list:
    comp_rows.append({
        "Model": m["Model"],
        "Target": target,
        "R2": round(float(m["R2"]), 4) if m.get("R2") is not None else None,
        "MAE": round(float(m["MAE"]), 4),
        "MSE": round(float(m["MSE"]), 4),
        "RMSE": round(float(m["RMSE"]), 4),
        "Notes": m.get("Feature") if m.get("Model")=="SLR" else ""
    })

if 'arima_metrics' in locals():
    comp_rows.append({
        "Model": arima_metrics["Model"],
        "Target": target,
        "R2": "N/A",
        "MAE": round(float(arima_metrics["MAE"]), 4),
        "MSE": round(float(arima_metrics["MSE"]), 4),
        "RMSE": round(float(arima_metrics["RMSE"]), 4),
        "Notes": f"AIC={round(float(arima_metrics['AIC']),2)}"
    })

comp_df = pd.DataFrame(comp_rows)

# Highlight best by RMSE
best_idx = comp_df["RMSE"].astype(float).idxmin()
def highlight_best(row):
    return ['background-color: lightgreen' if row.name == best_idx else '' for _ in row]

st.dataframe(comp_df.style.apply(highlight_best, axis=1))

# ======================================
# Explanatory Section
# ======================================
with st.expander("ℹ️ Regression vs ARIMA — When to use what?", expanded=False):
    st.markdown(
    """
    **Regression (SLR/MLR)**  
    - Uses tabular features (OHLC, Volume, Sector) to explain current target values.  
    - Best for *explanatory modeling* and cross-sectional prediction.  

    **ARIMA**  
    - Uses only the past values of the target to forecast future points.  
    - Best for *time-series forecasting* when autocorrelation exists.  
    """
    )
