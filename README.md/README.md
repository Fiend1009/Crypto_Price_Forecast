📈 Crypto Price Forecast Dashboard
An interactive Streamlit dashboard for visualizing cryptocurrency prices, technical indicators, and forecasts using Prophet, LSTM, and a Hybrid Prophet + LSTM model.

🚀 Features
Live cryptocurrency price charts with candlesticks.

Technical indicators: RSI, MACD, and moving averages.

Three forecast models:

Prophet only

LSTM only

Hybrid (Prophet + LSTM combined)

Multi-coin support:

Bitcoin (BTC)

Ethereum (ETH)

Solana (SOL)

Dogecoin (DOGE)

Error metrics tab for model performance comparison.

🛠 Tech Stack
Python (pandas, numpy, plotly, ta, tensorflow, prophet)

Streamlit for the interactive dashboard

Plotly for charts

Prophet for time-series forecasting

LSTM via TensorFlow/Keras

📦 Installation & Setup
1️⃣ Clone the repository
bash
Copy
Edit
git clone https://github.com/YOUR_USERNAME/Crypto_Price_Forecast.git
cd Crypto_Price_Forecast
2️⃣ Create a virtual environment & install dependencies
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

pip install -r requirements.txt
3️⃣ Run locally
bash
Copy
Edit
streamlit run dashboard/app.py
🌐 Live Dashboard
You can try the app here:
🔗 Crypto Price Forecast Dashboard

📸 Screenshots
Price Chart	Forecast Comparison

📌 Save screenshots from your dashboard in a screenshots/ folder and replace these placeholder images.

📊 Forecast Models
Prophet → Captures seasonality & trends.

LSTM → Learns complex sequential patterns.

Hybrid → Combines Prophet & LSTM predictions for improved accuracy.

📜 License
This project is licensed under the MIT License.