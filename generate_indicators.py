import pandas as pd
from pathlib import Path

# === Settings ===
DATA_DIR = Path("data")
COINS = ["bitcoin", "ethereum", "solana", "dogecoin"]

# === Functions ===

def compute_rsi(series, window=14):
    """Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def process_coin(coin):
    input_path = DATA_DIR / f"{coin}.csv"
    output_path = DATA_DIR / f"{coin}_indicators.csv"

    if not input_path.exists():
        print(f"❌ Missing data file for {coin}: {input_path}")
        return

    # Load Data
    df = pd.read_csv(input_path, parse_dates=["Date"])
    
    # Ensure numeric columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # === 1. Moving Averages ===
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA21'] = df['Close'].rolling(window=21).mean()

    # === 2. RSI ===
    df['RSI14'] = compute_rsi(df['Close'], 14)

    # === 3. MACD (matching dashboard names) ===
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_12_26_9'] = ema12 - ema26
    df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
    df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']

    # === 4. Bollinger Bands ===
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

    # === 5. ATR (Average True Range) ===
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR14'] = true_range.rolling(window=14).mean()

    # Save file
    df.to_csv(output_path, index=False)
    print(f"✔ Saved: {output_path}")


# === Process All Coins ===
for coin in COINS:
    process_coin(coin)

print("✅ All indicators generated successfully!")
