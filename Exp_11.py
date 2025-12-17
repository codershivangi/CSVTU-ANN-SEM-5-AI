# To study the use of Long Short Term Memory/Gated Recurrent Units to predict the stock prices based on historical data.

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense

df = yf.download("AAPL", period="1y")
data = df[['Close']].values

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

X, y = [], []
for i in range(60, len(scaled)):
    X.append(scaled[i-60:i])
    y.append(scaled[i])
X, y = np.array(X), np.array(y)

model_lstm = Sequential([LSTM(50, return_sequences=False, input_shape=(60,1)), Dense(1)])
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X, y, epochs=3, batch_size=16)

model_gru = Sequential([GRU(50, return_sequences=False, input_shape=(60,1)), Dense(1)])
model_gru.compile(optimizer='adam', loss='mse')
model_gru.fit(X, y, epochs=3, batch_size=16)

print("Training complete.")
