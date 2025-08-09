# scripts/generate_lstm_all.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

COINS = ["bitcoin", "ethereum", "solana", "dogecoin"]
DATA_DIR = "data"
LOOKBACK = 60
FORECAST_DAYS = 30
EPOCHS = 10   # increase if you want better training
BATCH = 32

for coin in COINS:
    folder = os.path.join(DATA_DIR, coin)
    indicators_path = os.path.join(folder, "indicators.csv")
    prophet_path = os.path.join(folder, "prophet_forecast.csv")
    if not os.path.exists(indicators_path):
        print(f"[skip] indicators missing for {coin}")
        continue

    df = pd.read_csv(indicators_path, parse_dates=["Date"])
    close = df["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    X, y = [], []
    for i in range(LOOKBACK, len(scaled)):
        X.append(scaled[i-LOOKBACK:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    if len(X) == 0:
        print(f"[skip] not enough data for {coin}")
        continue
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")

    print(f"Training LSTM for {coin}...")
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH, verbose=1)

    # iterative forecast
    last_input = scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)
    preds = []
    current = last_input.copy()
    for _ in range(FORECAST_DAYS):
        p = model.predict(current, verbose=0)
        preds.append(p[0, 0])
        current = np.append(current[:, 1:, :], [[p]], axis=1)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
    future_dates = pd.date_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=FORECAST_DAYS)

    out_df = pd.DataFrame({"Date": future_dates, "yhat": preds.flatten()})
    out_df.to_csv(os.path.join(folder, "lstm_forecast.csv"), index=False)
    print(f"[✔] LSTM forecast saved for {coin}")

    # Create hybrid if prophet exists
    if os.path.exists(prophet_path):
        prop = pd.read_csv(prophet_path, parse_dates=["Date"])
        prop_future = prop.tail(FORECAST_DAYS).reset_index(drop=True)
        if len(prop_future) == FORECAST_DAYS:
            hybrid_yhat = (prop_future["yhat"].values + preds.flatten()) / 2
            hybrid_df = pd.DataFrame({"Date": future_dates, "yhat": hybrid_yhat})
            hybrid_df.to_csv(os.path.join(folder, "hybrid_forecast.csv"), index=False)
            print(f"[✔] Hybrid forecast saved for {coin}")
        else:
            print(f"[!] Prophet forecast length mismatch for {coin}; hybrid not created.")
