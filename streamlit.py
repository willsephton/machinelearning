import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from prophet import Prophet
import numpy as np
import seaborn as sns
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima_model import ARIMA
from pmdarima import auto_arima

# Data Retrieval
with open("dataset/list_of_tickers.txt", "r") as file:
    tickers = file.read().splitlines()

def gatherStockDataForProphet(tickers):
    stockData = yf.download(tickers, period='1y', interval='1d')['Close']
    return stockData

def prophetFunction(chosen_ticker, days, prophetData):
    data = prophetData.reset_index().rename(columns={'Date': 'ds', chosen_ticker: 'y'})
    data['y'] = np.log(data['y'])
    model = Prophet(
        daily_seasonality=True,
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )

    model.fit(data)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    signals = createSignals(forecast)
    st.write(f"Signal for the forecast period: {signals}")

    st.pyplot(model.plot(forecast))

def createSignals(forecast):
    buy_threshold = 0.1
    sell_threshold = -0.1
    
    buy_count = sum(forecast['yhat'].pct_change() > buy_threshold)
    sell_count = sum(forecast['yhat'].pct_change() < sell_threshold)
    
    if buy_count > sell_count:
        return 'Buy'
    else:
        return 'Sell'
    
  
def arimaFunction(specficStock, chosen_ticker):
        # Fit auto_arima
    model = auto_arima(specficStock, start_p=1, start_q=1,
                    max_p=3, max_q=3, m=12,
                    start_P=0, seasonal=True,
                    d=1, D=1, trace=True,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True)

    # Summary and best parameters
    print(model.summary())
    print(model.get_params())

    # Forecast
    n_periods = 10
    forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)

    # Plot Forecast
    plt.figure(figsize=(10, 6))
    plt.plot(specficStock.index, specficStock, label='Original Data')
    plt.plot(pd.date_range(start=specficStock.index[-1], periods=n_periods + 1, freq='M')[1:], forecast, color='red', label='Forecast')
    plt.fill_between(pd.date_range(start=specficStock.index[-1], periods=n_periods + 1, freq='M')[1:], conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'ARIMA Forecast for {chosen_ticker}')
    plt.legend()
    st.pyplot()


def main():
    st.title('Stock Analysis App')

    chosen_tickers = ["META", "AVGO", "BKNG", "TSLA"]
    chosen_ticker = st.sidebar.selectbox('Select a Ticker', chosen_tickers)

    option = st.sidebar.selectbox(
        'Select an Option',
        ('Prophet Forecast', 'ARIMA Forecast', 'LSTM Forecast')
    )

    stockData = gatherStockDataForProphet(chosen_tickers)

    if option == 'Prophet Forecast':
        st.subheader('Prophet Forecast')
        days = st.slider('Select Forecasting Period (in days)', min_value=1, max_value=365, value=30)
        prophetFunction(chosen_ticker, days, stockData)

    if option == 'ARIMA Forecast':
        st.subheader('Arima Forecast')
        arimaData = gatherStockDataForProphet(tickers)
        specficStock = arimaData[chosen_ticker]
        arimaFunction(specficStock, chosen_ticker)

if __name__ == "__main__":
    main()
