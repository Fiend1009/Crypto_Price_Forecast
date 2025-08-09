# scripts/fetch_data.py
import yfinance as yf
import os
import pandas as pd

COINS = {
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "solana": "SOL-USD",
    "dogecoin": "DOGE-USD"
}

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

def fetch_coin(coin, ticker):
    folder = os.path.join(OUT_DIR, coin)
    os.makedirs(folder, exist_ok=True)
    print(f"Fetching {coin} ({ticker})...")
    df = yf.download(ticker, start="2020-01-01", progress=False)
    df.reset_index(inplace=True)
    df.to_csv(os.path.join(folder, f"{coin}.csv"), index=False)
    print(f"Saved {coin} -> {os.path.join(folder, f'{coin}.csv')}")

if __name__ == "__main__":
    for coin, ticker in COINS.items():
        fetch_coin(coin, ticker)
    print("All coins fetched.")
