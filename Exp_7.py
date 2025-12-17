# To study Long Short Term Memory for Time Series Prediction.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Generate sine wave data
def generate_sine(n=1500):
    x = np.arange(n)
    return np.sin(0.01 * x)
data = generate_sine()

# Scale data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.reshape(-1,1))

# Create sequences
def create_seq(series, seq_len):
    X, y = [], []
    for i in range(len(series)-seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    return np.array(X), np.array(y)

SEQ_LEN = 50
X, y = create_seq(data_scaled, SEQ_LEN)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = Sequential([
LSTM(64, input_shape=(SEQ_LEN,1)),
Dropout(0.2),
Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Predict
pred = model.predict(X_test)
pred_inv = scaler.inverse_transform(pred)
y_test_inv = scaler.inverse_transform(y_test)

print("Prediction completed.")
