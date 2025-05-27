import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load model
model = load_model('Stock Predictions Model.keras', compile=False)


st.header('📈 Stock Market Predictor')

# Input: Stock Symbol
stock = st.text_input('Enter Stock Symbol', 'GOOG')

# Date range
start = '2012-01-01'
end = '2027-12-31'

# Download stock data
data = yf.download(stock, start, end)

st.subheader('🔍 Raw Stock Data')
st.write(data)

# Train-Test Split
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

# Scale Data
scaler = MinMaxScaler(feature_range=(0,1))
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Moving Averages
st.subheader('📉 Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(data.Close, 'g', label='Closing Price')
plt.legend()
st.pyplot(fig1)

st.subheader('📉 Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r', label='MA50')
plt.plot(ma_100_days, 'b', label='MA100')
plt.plot(data.Close, 'g', label='Closing Price')
plt.legend()
st.pyplot(fig2)

st.subheader('📉 Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r', label='MA100')
plt.plot(ma_200_days, 'b', label='MA200')
plt.plot(data.Close, 'g', label='Closing Price')
plt.legend()
st.pyplot(fig3)

# Model Testing
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)
predict = model.predict(x)

scale = 1 / scaler.scale_
predict = predict * scale
y = y * scale

# Plot Actual vs Predicted
st.subheader('✅ Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(y, 'g', label='Original Price')
plt.plot(predict, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)

# ⏩ User Input: Future Prediction Days
st.subheader("🔮 Predict Future Stock Prices")
future_days = st.number_input("Enter number of future days to predict", min_value=1, max_value=100, value=50, step=1)

if st.button('Predict Future'):
    # Prepare last 100 days data for prediction
    last_100_days = data_test_scale[-100:]
    future_input = last_100_days.reshape(1, 100, 1)

    future_predictions = []

    for _ in range(future_days):
        pred = model.predict(future_input)[0][0]
        future_predictions.append(pred)

        # Append and reshape
        future_input = np.append(future_input[:, 1:, :], [[[pred]]], axis=1)

    # Reverse scale predictions
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = future_predictions * scale

    # Plot predictions
    st.subheader(f'📅 Next {future_days} Days Forecast')
    fig5 = plt.figure(figsize=(10,6))
    plt.plot(future_predictions, 'b', label='Future Predicted Prices')
    plt.xlabel('Days Ahead')
    plt.ylabel('Predicted Price')
    plt.legend()
    plt.grid()
    st.pyplot(fig5)

    # Optional: Show table
    future_df = pd.DataFrame(future_predictions, columns=['Predicted Price'])
    st.write(future_df)
