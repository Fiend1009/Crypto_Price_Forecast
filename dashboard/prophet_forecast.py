# prophet_forecast.py
import pandas as pd
from prophet import Prophet

# === Load Data ===
df = pd.read_csv("data/bitcoin_with_indicators.csv", parse_dates=["Date"])
df = df[["Date", "Close"]].dropna()

# Prepare for Prophet
prophet_df = df.rename(columns={"Date": "ds", "Close": "y"})

# Train Prophet Model
prophet = Prophet(daily_seasonality=True, yearly_seasonality=True)
prophet.fit(prophet_df)

# Forecast for next 30 days
future = prophet.make_future_dataframe(periods=30)
forecast = prophet.predict(future)

# Save results
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv("data/prophet_forecast.csv", index=False)
print("✔ Prophet forecast saved to data/prophet_forecast.csv")
