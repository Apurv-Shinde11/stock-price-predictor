import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, Input, Concatenate
from datetime import datetime, timedelta
import requests
import matplotlib.pyplot as plt
from io import BytesIO

# --- Backend functions ---

def download_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, start=start_date, end=end_date, threads=False)
        if df.empty:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(start=start_date, end=end_date)
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        return df
    except Exception as e:
        st.error(f"Error downloading stock data: {e}")
        return pd.DataFrame()

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["Close"])
    df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
    df['EMA_10'] = ta.trend.ema_indicator(df['Close'], window=10)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    macd = ta.trend.macd(df['Close'])
    df['MACD'] = macd
    df['MACD_signal'] = ta.trend.macd_signal(df['Close'])
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    df = df.dropna()
    return df

def preprocess_data(df: pd.DataFrame, feature_cols: list, target_col: str, sequence_length=60):
    df = df.dropna(subset=feature_cols + [target_col])
    data = df[feature_cols + [target_col]].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i-sequence_length:i, :-1])
        y.append(data_scaled[i, -1])
    return np.array(X), np.array(y), scaler

def build_lstm_attention_model(input_shape):
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(50, return_sequences=True)(inputs)
    attention_out = Attention()([lstm_out, lstm_out])
    lstm_out2 = LSTM(50)(attention_out)
    dropout = Dropout(0.2)(lstm_out2)
    output = Dense(1)(dropout)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future_days(model, last_sequence, scaler, feature_cols, days=5):
    predictions_scaled = []
    current_seq = last_sequence.copy()
    close_idx = feature_cols.index('Close')
    for _ in range(days):
        pred_scaled = model.predict(current_seq[np.newaxis, :, :])[0, 0]
        predictions_scaled.append(pred_scaled)
        new_row = np.zeros(len(feature_cols))
        new_row[close_idx] = pred_scaled
        current_seq = np.vstack([current_seq[1:], new_row])
    dummy_input = np.zeros((len(predictions_scaled), len(feature_cols) + 1))
    dummy_input[:, -1] = predictions_scaled
    predicted_prices = scaler.inverse_transform(dummy_input)[:, -1]
    return predicted_prices

def fetch_finnhub_sentiment(ticker, token):
    base_url = "https://finnhub.io/api/v1/news-sentiment"
    try:
        response = requests.get(base_url, params={"symbol": ticker, "token": token})
        if response.status_code == 200:
            data = response.json()
            score = data.get("sentiment", {}).get("regularMarketChangePercent", 0)
            return score
        else:
            return 0
    except Exception:
        return 0

def plot_predictions(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(y_true, label='Actual Close Price')
    ax.plot(y_pred, label='Predicted Close Price')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Price')
    ax.set_title('Stock Price Prediction')
    ax.legend()
    st.pyplot(fig)

# --- Streamlit UI ---

st.title("AlphaPilot: Stock Price Predictor with Sentiment")

ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, RELIANCE.NS)", value="AAPL").upper().strip()
start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
end_date = datetime.now().date()
st.write(f"End Date (fixed to today): {end_date}")

if st.button("Run Prediction"):
    with st.spinner("Downloading and processing data..."):
        df = download_stock_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        if df.empty:
            st.error("No data found for this ticker and date range.")
            st.stop()

        df = add_technical_indicators(df)

        feature_cols = ['Close', 'SMA_10', 'EMA_10', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower']
        target_col = 'Close'
        sequence_length = 60

        X, y, scaler = preprocess_data(df, feature_cols, target_col, sequence_length)
        if len(X) == 0:
            st.error("Not enough data after preprocessing.")
            st.stop()

        input_shape = (X.shape[1], X.shape[2])
        model = build_lstm_attention_model(input_shape)

        history = model.fit(X, y, epochs=20, batch_size=32, validation_split=0.1, verbose=0)
        st.success("Model training complete!")

        # Predict on training data
        y_pred_scaled = model.predict(X)
        dummy_input = np.zeros((len(y_pred_scaled), len(feature_cols) + 1))
        dummy_input[:, -1] = y_pred_scaled.flatten()
        y_pred = scaler.inverse_transform(dummy_input)[:, -1]

        dummy_input[:, -1] = y
        y_true = scaler.inverse_transform(dummy_input)[:, -1]

        plot_predictions(y_true, y_pred)

        last_seq = X[-1]
        future_days = st.slider("Select number of future days to predict", 1, 10, 5)
        future_prices = predict_future_days(model, last_seq, scaler, feature_cols, days=future_days)

        st.write(f"Predicted Close Prices for next {future_days} days:")
        for i, price in enumerate(future_prices, 1):
            st.write(f"Day {i}: ${price:.2f}")

        # Sentiment analysis
        finnhub_token = st.secrets.get("d0pd9j1r01qr8ds30s60d0pd9j1r01qr8ds30s6g", None)
        if not finnhub_token:
            st.warning("Finnhub API token not found in secrets. Please add it for sentiment analysis.")
        else:
            sentiment_score = fetch_finnhub_sentiment(ticker, finnhub_token)
            st.write(f"Sentiment Score (Finnhub): {sentiment_score:.3f}")

st.markdown(
    """
    ---
    Developed by Apurv | Powered by TensorFlow, yfinance, Finnhub & Streamlit
    """
)