# scripts/generate_all_forecasts.py
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# Prophet
from prophet import Prophet

# LSTM related
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ---------------------------
# CONFIG
# ---------------------------
COINS = ["bitcoin", "ethereum", "solana", "dogecoin"]   # extend if needed
DATA_DIR = Path("data")
LOOKBACK = 60            # LSTM lookback window (days)
FORECAST_DAYS = 30       # horizon for all models
LSTM_EPOCHS = 10         # set low for testing; increase for better accuracy
LSTM_BATCH = 32

# ---------------------------
# HELPERS
# ---------------------------
def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def fit_prophet_and_forecast(df, horizon_days=30):
    """
    df: DataFrame with Date (datetime) and Close
    Returns DataFrame with Date & yhat for the next horizon_days
    """
    dfp = df[["Date", "Close"]].dropna().rename(columns={"Date": "ds", "Close": "y"})
    if len(dfp) < 30:
        raise ValueError("Not enough data for Prophet (need >=30 rows).")

    m = Prophet(daily_seasonality=True)
    m.fit(dfp)
    future = m.make_future_dataframe(periods=horizon_days)
    forecast = m.predict(future)
    # select the future rows (strictly after last date in df)
    last_date = df["Date"].max()
    future_forecast = forecast[forecast["ds"] > last_date].head(horizon_days)[["ds", "yhat"]].reset_index(drop=True)
    future_forecast = future_forecast.rename(columns={"ds": "Date", "yhat": "yhat"})
    return future_forecast

def train_lstm_and_forecast(df, lookback=60, horizon_days=30, epochs=10, batch=32):
    """
    df: DataFrame with Date (datetime) and Close
    Returns DataFrame with Date & yhat for the next horizon_days
    """
    series = df[["Date", "Close"]].dropna().reset_index(drop=True)
    if len(series) < lookback + 1:
        raise ValueError(f"Not enough data for LSTM (need >={lookback+1} rows).")

    close = series["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # build model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")

    model.fit(X, y, epochs=epochs, batch_size=batch, verbose=1)

    # iterative forecast: seed with last `lookback` values
    last_seq = scaled[-lookback:].reshape(1, lookback, 1)
    preds_scaled = []
    current = last_seq.copy()
    for _ in range(horizon_days):
        p = model.predict(current, verbose=0)[0, 0]
        preds_scaled.append(p)
        # append and shift
        current = np.append(current[:, 1:, :], [[[p]]], axis=1)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    start_date = series["Date"].iloc[-1] + pd.Timedelta(days=1)
    dates = pd.date_range(start=start_date, periods=horizon_days, freq="D")
    out = pd.DataFrame({"Date": dates, "yhat": preds})
    return out

# ---------------------------
# MAIN: loop coins
# ---------------------------
if __name__ == "__main__":
    for coin in COINS:
        print(f"\n--- Processing {coin} ---")
        coin_folder = DATA_DIR / coin
        indicators_path = coin_folder / "indicators.csv"

        if not indicators_path.exists():
            print(f"[skip] indicators file not found for {coin} -> {indicators_path}")
            continue

        try:
            df_ind = pd.read_csv(indicators_path, parse_dates=["Date"])
        except Exception as e:
            print(f"[error] failed reading {indicators_path}: {e}")
            continue

        # ensure numeric
        df_ind = ensure_numeric(df_ind, ["Open", "High", "Low", "Close", "Volume", "Adj Close"])
        df_ind.dropna(subset=["Date", "Close"], inplace=True)
        df_ind.sort_values("Date", inplace=True)
        df_ind.reset_index(drop=True, inplace=True)

        # Prophet
        try:
            prophet_forecast = fit_prophet_and_forecast(df_ind, horizon_days=FORECAST_DAYS)
            prophet_out_path = coin_folder / "prophet_forecast.csv"
            prophet_forecast.to_csv(prophet_out_path, index=False)
            print(f"[✔] Prophet forecast saved: {prophet_out_path}")
        except Exception as e:
            print(f"[!] Prophet failed for {coin}: {e}")
            prophet_forecast = None

        # LSTM
        try:
            lstm_forecast = train_lstm_and_forecast(df_ind, lookback=LOOKBACK, horizon_days=FORECAST_DAYS, epochs=LSTM_EPOCHS, batch=LSTM_BATCH)
            lstm_out_path = coin_folder / "lstm_forecast.csv"
            lstm_forecast.to_csv(lstm_out_path, index=False)
            print(f"[✔] LSTM forecast saved: {lstm_out_path}")
        except Exception as e:
            print(f"[!] LSTM failed for {coin}: {e}")
            lstm_forecast = None

        # Hybrid: average Prophet & LSTM (only if both exist and dates align)
        try:
            if (prophet_forecast is not None) and (lstm_forecast is not None):
                # align by Date
                pf = prophet_forecast.copy().reset_index(drop=True)
                lf = lstm_forecast.copy().reset_index(drop=True)

                # If prophet returned exactly FORECAST_DAYS future dates (most likely), use those.
                # Otherwise merge on Date (inner join).
                if len(pf) == FORECAST_DAYS and len(lf) == FORECAST_DAYS:
                    hybrid_yhat = (pf["yhat"].values + lf["yhat"].values) / 2.0
                    hybrid_df = pd.DataFrame({"Date": pf["Date"], "yhat": hybrid_yhat})
                else:
                    merged = pd.merge(pf, lf, on="Date", how="inner", suffixes=("_prop", "_lstm"))
                    if merged.empty:
                        raise ValueError("No overlapping forecast dates to compute hybrid.")
                    hybrid_yhat = (merged["yhat_prop"].values + merged["yhat_lstm"].values) / 2.0
                    hybrid_df = pd.DataFrame({"Date": merged["Date"], "yhat": hybrid_yhat})

                hybrid_out_path = coin_folder / "hybrid_forecast.csv"
                hybrid_df.to_csv(hybrid_out_path, index=False)
                print(f"[✔] Hybrid forecast saved: {hybrid_out_path}")
            else:
                print(f"[!] Hybrid not created for {coin} (missing prophet or lstm forecast).")
        except Exception as e:
            print(f"[!] Hybrid creation failed for {coin}: {e}")

    print("\nAll done.")
