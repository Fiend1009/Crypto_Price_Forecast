# scripts/generate_prophet_all.py
import os
import pandas as pd
from prophet import Prophet

COINS = ["bitcoin", "ethereum", "solana", "dogecoin"]
DATA_DIR = "data"
FORECAST_DAYS = 30

for coin in COINS:
    folder = os.path.join(DATA_DIR, coin)
    indicators_path = os.path.join(folder, "indicators.csv")
    if not os.path.exists(indicators_path):
        print(f"[skip] indicators missing for {coin}")
        continue

    df = pd.read_csv(indicators_path, parse_dates=["Date"])
    df = df[["Date", "Close"]].dropna()
    df_prophet = df.rename(columns={"Date": "ds", "Close": "y"})

    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=FORECAST_DAYS)
    forecast = model.predict(future)

    out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "Date"})
    out.to_csv(os.path.join(folder, "prophet_forecast.csv"), index=False)
    print(f"[✔] Prophet forecast saved for {coin}")
