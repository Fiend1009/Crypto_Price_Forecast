import pandas as pd
from prophet import Prophet
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# === Load Data ===
df = pd.read_csv("data/bitcoin_with_indicators.csv", parse_dates=["Date"])
df = df[["Date", "Close"]].dropna()

# ========================
# STEP 1: Prophet Forecast
# ========================
prophet_df = df.rename(columns={"Date": "ds", "Close": "y"})
prophet = Prophet(daily_seasonality=True, yearly_seasonality=True)
prophet.fit(prophet_df)

future = prophet.make_future_dataframe(periods=30)
prophet_forecast = prophet.predict(future)
prophet_preds = prophet_forecast[["ds", "yhat"]].set_index("ds")

# ========================
# STEP 2: LSTM Forecast
# ========================
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

seq_len = 60
x_train, y_train = [], []
for i in range(seq_len, len(scaled_data)):
    x_train.append(scaled_data[i - seq_len:i, 0])
    y_train.append(scaled_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1)

# Prepare test data for LSTM
test_data = scaled_data[-seq_len:]
X_test = []
current_batch = test_data.reshape((1, seq_len, 1))
for _ in range(30):
    pred = model.predict(current_batch, verbose=0)[0]
    X_test.append(pred)
    current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)

lstm_preds = scaler.inverse_transform(np.array(X_test).reshape(-1, 1))
lstm_dates = pd.date_range(df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=30)
lstm_forecast = pd.DataFrame({"yhat": lstm_preds.flatten()}, index=lstm_dates)

# ========================
# STEP 3: Hybrid Model
# ========================
hybrid_forecast = (prophet_preds.tail(30) * 0.6) + (lstm_forecast * 0.4)
hybrid_forecast.columns = ["Hybrid_Prediction"]

# Save results
hybrid_forecast.to_csv("data/hybrid_forecast.csv")
print("✔ Hybrid Prophet + LSTM forecast saved.")
