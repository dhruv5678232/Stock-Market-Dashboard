import io
ax.set_xlabel("Predicted")
ax.set_ylabel("Residual")
st.pyplot(fig)


# 8) ARIMA on target series (requires order by time for best accuracy)
st.subheader("⏱️ ARIMA Forecasting")


arima_outputs = run_arima(y, test_size=0.2, forecast_horizon=int(forecast_horizon))
if arima_outputs is None:
st.warning("ARIMA could not be fit (series too short or no suitable (p,d,q) found). Ensure your target is time-indexed and has enough rows.")
else:
arima_metrics, pred_test, y_test, future_forecast, best_order = arima_outputs


# Plot test vs forecast
fig, ax = plt.subplots(figsize=(9,4))
# Align indexes if available
if isinstance(y.index, pd.DatetimeIndex):
test_index = y.iloc[-len(y_test):].index
ax.plot(test_index, y_test, label="Actual (test)")
ax.plot(test_index, pred_test, label=f"Predicted ARIMA{best_order}")
else:
ax.plot(range(len(y_test)), y_test.values, label="Actual (test)")
ax.plot(range(len(pred_test)), pred_test.values, label=f"Predicted ARIMA{best_order}")
ax.set_title("ARIMA — Test Period Forecast vs Actual")
ax.legend()
st.pyplot(fig)


# Future forecast plot
fig, ax = plt.subplots(figsize=(9,4))
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


# 9) Comparison table
st.subheader("📋 Model Comparison")
comp_rows = []
for m in metrics_list:
comp_rows.append({
"Model": m.get("Model"),
"Target": target,
"R2": None if m.get("R2") is None else round(float(m.get("R2")), 4),
"MAE": round(float(m.get("MAE")), 4),
"MSE": round(float(m.get("MSE")), 4),
"RMSE": round(float(m.get("RMSE")), 4),
"Notes": m.get("Feature") if m.get("Model","").startswith("SLR") else ""
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
st.dataframe(comp_df)


# 10) Explanatory section
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