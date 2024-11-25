import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import timedelta

# App title
st.title("Apple Stock Price Prediction")
st.write("Predict the Apple stock prices for the next 30 days using historical data.")

# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(data.head())
    
    # Data exploration
    st.write("Summary Statistics:")
    st.write(data.describe())
    
    # Visualization
    st.line_chart(data[['Close']])
    
    # Preprocessing
    st.write("Preprocessing the data...")
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    close_prices = data['Close'].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices.reshape(-1, 1))
    
    # Prepare training data
    sequence_length = 60
    X_train, y_train = [], []
    for i in range(sequence_length, len(scaled_data)):
        X_train.append(scaled_data[i-sequence_length:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # Model building
    st.write("Training the model...")
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(units=50, return_sequences=False),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=1)
    
    # Prediction
    st.write("Making predictions for the next 30 days...")
    test_data = scaled_data[-sequence_length:]
    predictions = []
    for _ in range(30):
        input_data = np.reshape(test_data[-sequence_length:], (1, sequence_length, 1))
        predicted_price = model.predict(input_data)[0, 0]
        predictions.append(predicted_price)
        test_data = np.append(test_data, predicted_price)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    # Visualization
    st.write("Prediction Results:")
    future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, 31)]
    prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': predictions.flatten()})
    st.write(prediction_df)
    
    fig, ax = plt.subplots()
    ax.plot(data.index[-100:], close_prices[-100:], label="Actual")
    ax.plot(prediction_df['Date'], prediction_df['Predicted Close'], label="Predicted", color='red')
    ax.set_title("Stock Price Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    st.pyplot(fig)
