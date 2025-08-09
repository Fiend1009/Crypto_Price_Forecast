# scripts/generate_indicators_all.py
import os
import pandas as pd
from pathlib import Path

COINS = ["bitcoin", "ethereum", "solana", "dogecoin"]
DATA_DIR = Path("data")

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta).clip(lower=0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

for coin in COINS:
    folder = DATA_DIR / coin
    raw_path = folder / f"{coin}.csv"
    out_path = folder / "indicators.csv"

    if not raw_path.exists():
        print(f"[skip] {raw_path} not found")
        continue

    df = pd.read_csv(raw_path, parse_dates=["Date"])

    # Convert to numeric robustly
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["Close"], inplace=True)

    # Moving averages
    df["MA7"] = df["Close"].rolling(window=7).mean()
    df["MA21"] = df["Close"].rolling(window=21).mean()

    # RSI14
    df["RSI14"] = compute_rsi(df["Close"], window=14)

    # MACD (12,26) and 9-signal
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD_12_26_9"] = ema12 - ema26
    df["MACDs_12_26_9"] = df["MACD_12_26_9"].ewm(span=9, adjust=False).mean()
    df["MACDh_12_26_9"] = df["MACD_12_26_9"] - df["MACDs_12_26_9"]

    # Bollinger Bands (20)
    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + 2 * bb_std
    df["BB_Lower"] = df["BB_Middle"] - 2 * bb_std

    # ATR14
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(window=14).mean()

    df.to_csv(out_path, index=False)
    print(f"[✔] Indicators saved for {coin}: {out_path}")
