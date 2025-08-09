# dashboard/app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

COINS = ["bitcoin", "ethereum", "solana", "dogecoin"]
DATA_DIR = "data"

st.set_page_config(page_title="Crypto Multi-Coin Forecast", layout="wide")
st.title("📈 Crypto Forecasting: Prophet / LSTM / Hybrid")

coin = st.sidebar.selectbox("Select coin", COINS)
model_choice = st.sidebar.radio("Model", ["Prophet", "LSTM", "Hybrid"])

coin_folder = os.path.join(DATA_DIR, coin)
ind_path = os.path.join(coin_folder, "indicators.csv")
prophet_path = os.path.join(coin_folder, "prophet_forecast.csv")
lstm_path = os.path.join(coin_folder, "lstm_forecast.csv")
hybrid_path = os.path.join(coin_folder, "hybrid_forecast.csv")

if not os.path.exists(ind_path):
    st.error(f"Missing indicators for {coin}. Run generate_indicators_all.py.")
    st.stop()

df = pd.read_csv(ind_path, parse_dates=["Date"])

def load_forecast(path):
    if os.path.exists(path):
        f = pd.read_csv(path, parse_dates=["Date"])
        return f
    return None

prophet = load_forecast(prophet_path)
lstm = load_forecast(lstm_path)
hybrid = load_forecast(hybrid_path)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Historical", line=dict(color="white")))

if model_choice == "Prophet" and prophet is not None:
    if "yhat" in prophet.columns:
        fig.add_trace(go.Scatter(x=prophet["Date"], y=prophet["yhat"], mode="lines", name="Prophet", line=dict(color="blue")))
elif model_choice == "LSTM" and lstm is not None:
    if "yhat" in lstm.columns:
        fig.add_trace(go.Scatter(x=lstm["Date"], y=lstm["yhat"], mode="lines", name="LSTM", line=dict(color="orange")))
elif model_choice == "Hybrid" and hybrid is not None:
    if "yhat" in hybrid.columns:
        fig.add_trace(go.Scatter(x=hybrid["Date"], y=hybrid["yhat"], mode="lines", name="Hybrid", line=dict(color="green")))
else:
    st.info("Selected forecast not available for this coin. Generate forecasts first (scripts).")

fig.update_layout(title=f"{coin.capitalize()} — {model_choice}", xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Indicators snapshot (latest)")
latest = df.iloc[-1]
cols_to_show = ["Close", "MA7", "MA21", "RSI14", "MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9", "BB_Upper", "BB_Lower", "ATR14"]
metrics = {c: (round(latest[c], 4) if c in latest and pd.notna(latest[c]) else "N/A") for c in cols_to_show}
col1, col2, col3 = st.columns(3)
for i, (k, v) in enumerate(metrics.items()):
    if i % 3 == 0:
        col = col1
    elif i % 3 == 1:
        col = col2
    else:
        col = col3
    col.write(f"**{k}**: {v}")

st.subheader("Model comparison (last 30 days)")
actual_last = df.tail(30).reset_index(drop=True)["Close"]

def compute_metrics_for(forecast_df):
    if forecast_df is None:
        return None
    if "yhat" in forecast_df.columns:
        preds = forecast_df.tail(30)["yhat"].reset_index(drop=True)
    elif "Forecast" in forecast_df.columns:
        preds = forecast_df.tail(30)["Forecast"].reset_index(drop=True)
    else:
        preds = forecast_df.tail(30).iloc[:, -1].reset_index(drop=True)
    if len(preds) != len(actual_last):
        if len(preds) > len(actual_last):
            preds = preds.tail(len(actual_last)).reset_index(drop=True)
        else:
            return None
    mae = mean_absolute_error(actual_last, preds)
    rmse = np.sqrt(mean_squared_error(actual_last, preds))
    mape = np.mean(np.abs((actual_last - preds) / actual_last)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

from sklearn.metrics import mean_absolute_error, mean_squared_error

prop_metrics = compute_metrics_for(prophet)
lstm_metrics = compute_metrics_for(lstm)
hybrid_metrics = compute_metrics_for(hybrid)

mrows = []
if prop_metrics:
    mrows.append({"Model": "Prophet", **prop_metrics})
if lstm_metrics:
    mrows.append({"Model": "LSTM", **lstm_metrics})
if hybrid_metrics:
    mrows.append({"Model": "Hybrid", **hybrid_metrics})

if mrows:
    metrics_df = pd.DataFrame(mrows)
    st.dataframe(metrics_df.style.format({"MAE":"{:.2f}", "RMSE":"{:.2f}", "MAPE":"{:.2f}"}))
else:
    st.info("Not enough matching forecast history to compute metrics. Make sure forecast CSVs contain >=30 forecast rows.")



# import streamlit as st
# import pandas as pd
# import plotly.graph_objects as go
# import os

# # Folder where data is stored
# DATA_DIR = "data"

# # List of available coins (matching your indicators.py output)
# COINS = ["bitcoin", "ethereum", "solana", "dogecoin"]

# st.set_page_config(page_title="Crypto Price Dashboard", layout="wide")

# st.title("📊 Cryptocurrency Technical Dashboard")

# # Sidebar - Coin Selection
# coin = st.sidebar.selectbox("Select Cryptocurrency", COINS)
# file_path = os.path.join(DATA_DIR, f"{coin}_indicators.csv")

# # Load data
# df = pd.read_csv(file_path, parse_dates=["Date"])

# # Main price chart with candlesticks + moving averages
# fig_price = go.Figure()

# fig_price.add_trace(go.Candlestick(
#     x=df["Date"],
#     open=df["Open"],
#     high=df["High"],
#     low=df["Low"],
#     close=df["Close"],
#     name="Price"
# ))

# fig_price.add_trace(go.Scatter(
#     x=df["Date"], y=df["MA7"], mode="lines", name="MA7", line=dict(color="orange")
# ))
# fig_price.add_trace(go.Scatter(
#     x=df["Date"], y=df["MA21"], mode="lines", name="MA21", line=dict(color="blue")
# ))

# fig_price.update_layout(title=f"{coin.capitalize()} Price Chart",
#                         yaxis_title="Price (USD)",
#                         xaxis_rangeslider_visible=False,
#                         template="plotly_dark")

# # RSI Chart
# fig_rsi = go.Figure()
# fig_rsi.add_trace(go.Scatter(
#     x=df["Date"], y=df["RSI14"], mode="lines", name="RSI14", line=dict(color="purple")
# ))
# fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
# fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
# fig_rsi.update_layout(title="Relative Strength Index (RSI)",
#                       yaxis_title="RSI", template="plotly_dark")

# # MACD Chart
# fig_macd = go.Figure()
# fig_macd.add_trace(go.Scatter(
#     x=df["Date"], y=df["MACD_12_26_9"], mode="lines", name="MACD", line=dict(color="cyan")
# ))
# fig_macd.add_trace(go.Scatter(
#     x=df["Date"], y=df["MACDs_12_26_9"], mode="lines", name="Signal", line=dict(color="orange")
# ))
# fig_macd.add_trace(go.Bar(
#     x=df["Date"], y=df["MACDh_12_26_9"], name="Histogram", marker_color="gray"
# ))
# fig_macd.update_layout(title="MACD",
#                        yaxis_title="MACD", template="plotly_dark")

# # Display in Streamlit
# st.plotly_chart(fig_price, use_container_width=True)
# st.plotly_chart(fig_rsi, use_container_width=True)
# st.plotly_chart(fig_macd, use_container_width=True)
