import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

# ==============================
# 🎯 Streamlit Title
# ==============================
st.title("📊 Stock Price Prediction — SLR, MLR & ARIMA")

# ==============================
# 📂 Load Dataset
# ==============================
file_path = r"C:\\Users\\Dhruv Patel\\OneDrive\\Desktop\\proj\\synthetic_stock_data.csv"
df = pd.read_csv(file_path)

# Clean & format
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
df = df.sort_values(by='Date')

st.write("### Dataset Preview", df.head())

# ==============================
# 🟦 SIMPLE LINEAR REGRESSION (SLR)
# ==============================
X_slr = df[['Open']]
y = df['Close']

X_train, X_test, y_train, y_test, date_train, date_test = train_test_split(
    X_slr, y, df['Date'], test_size=0.2, shuffle=False
)

slr_model = LinearRegression()
slr_model.fit(X_train, y_train)
y_pred_slr = slr_model.predict(X_test)

r2_slr = r2_score(y_test, y_pred_slr)
mse_slr = mean_squared_error(y_test, y_pred_slr)

st.subheader("🔹 Simple Linear Regression (Close ~ Open)")
st.write(f"R²: {r2_slr:.4f}, MSE: {mse_slr:.2f}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(date_test, y_test, label="Actual", color="blue")
ax.plot(date_test, y_pred_slr, label="Predicted (SLR)", color="red")
ax.legend()
ax.set_title("SLR Prediction")
st.pyplot(fig)

# ==============================
# 🟨 MULTIPLE LINEAR REGRESSION (MLR)
# ==============================
features = ['Open', 'High', 'Low', 'Volume', 'Market_Cap',
            'PE_Ratio', 'Dividend_Yield', 'Volatility', 'Sentiment_Score']

X_mlr = df[features].dropna()
y = df.loc[X_mlr.index, 'Close']
dates = df.loc[X_mlr.index, 'Date']

X_train, X_test, y_train, y_test, date_train, date_test = train_test_split(
    X_mlr, y, dates, test_size=0.2, shuffle=False
)

mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)
y_pred_mlr = mlr_model.predict(X_test)

r2_mlr = r2_score(y_test, y_pred_mlr)
mse_mlr = mean_squared_error(y_test, y_pred_mlr)

st.subheader("🔹 Multiple Linear Regression (Close ~ Features)")
st.write(f"R²: {r2_mlr:.4f}, MSE: {mse_mlr:.2f}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(date_test, y_test, label="Actual", color="blue")
ax.plot(date_test, y_pred_mlr, label="Predicted (MLR)", color="green")
ax.legend()
ax.set_title("MLR Prediction")
st.pyplot(fig)

# ==============================
# 🟧 ARIMA (Time-Series Forecasting)
# ==============================
st.subheader("🔹 ARIMA Forecast on Close Price")

try:
    model = ARIMA(df['Close'], order=(5,1,0))  # can be tuned
    arima_fit = model.fit()
    
    forecast_steps = 30  # predict next 30 days
    forecast = arima_fit.get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=df['Date'].iloc[-1], periods=forecast_steps+1, freq='D')[1:]
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['Date'], df['Close'], label="Actual")
    ax.plot(forecast_index, forecast_mean, label="Forecast", color="orange")
    ax.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], 
                    color="orange", alpha=0.3)
    ax.legend()
    ax.set_title("ARIMA Forecast of Close Price")
    st.pyplot(fig)

except Exception as e:
    st.warning(f"ARIMA failed: {e}")

# ==============================
# 📋 Model Comparison
# ==============================
comparison = pd.DataFrame({
    "Model": ["SLR", "MLR", "ARIMA"],
    "R²": [r2_slr, r2_mlr, "N/A"],
    "MSE": [mse_slr, mse_mlr, "Check forecast error separately"]
})
st.write("### 📊 Model Comparison", comparison)
