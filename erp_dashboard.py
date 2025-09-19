import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load Forecasted Data
sarima = pd.read_csv("wine_forecast_sarima.csv")
rf_xgb = pd.read_csv("wine_forecast_rf_xgb.csv")
reorder = pd.read_csv("wine_reorder_alerts.csv")

# Evaluation Metrics
metrics = {
    'SARIMA': {'MAE': 86.91, 'RMSE': 93.21, 'MAPE': 0.126},
    'Random Forest': {'MAE': 84.31, 'RMSE': 92.43, 'MAPE': 0.114},
    'XGBoost': {'MAE': 82.47, 'RMSE': 90.97, 'MAPE': 0.110}
}

st.title("📦 ERP Dashboard - Seasonal Demand Preparedness")

# Section 1: Forecast Plot
st.subheader("📈 Forecast Comparison: SARIMA vs RF vs XGBoost")
fig, ax = plt.subplots(figsize=(12, 5))
sarima['Forecasted Sales'].plot(ax=ax, label='SARIMA')
rf_xgb['RandomForest Forecast'].plot(ax=ax, label='Random Forest')
rf_xgb['XGBoost Forecast'].plot(ax=ax, label='XGBoost')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Forecasted Sales')
plt.title('Wine Sales Forecast for Next 12 Months')
plt.grid(True)
st.pyplot(fig)

# Section 2: Evaluation Metrics
st.subheader("📊 Model Evaluation Metrics")
eval_df = pd.DataFrame(metrics).T
st.dataframe(eval_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE": "{:.2%}"}))

# Section 3: Reorder Alerts
st.subheader("🚨 Reorder Alerts Based on Safety Stock")
st.dataframe(reorder[reorder['Reorder Needed'] == True][['Forecasted Sales', 'Reorder Needed']])

# Section 4: Downloadable Reports
st.subheader("⬇️ Download Forecast CSVs")
with open("wine_forecast_sarima.csv", "rb") as f:
    st.download_button("Download SARIMA Forecast CSV", f, file_name="wine_forecast_sarima.csv")
with open("wine_forecast_rf_xgb.csv", "rb") as f:
    st.download_button("Download RF & XGBoost Forecast CSV", f, file_name="wine_forecast_rf_xgb.csv")
with open("wine_reorder_alerts.csv", "rb") as f:
    st.download_button("Download Reorder Alert CSV", f, file_name="wine_reorder_alerts.csv")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | ERP Project - Forecasting & Reordering")
