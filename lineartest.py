import pandas as pd
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import yfinance as yf

# Load or create your time series data

with open("dataset/list_of_tickers.txt", "r") as file:
    tickers = file.read().splitlines()
    
# Replace this with your own data loading method
def gatherStockDataArima():
    stockData = yf.download(tickers, period='1y', interval='1d')['Close']

    #arimaFormat = stockData.fillna(0) #Cleans rows with empty cells

    #arimaFormat.to_csv('arimaData.csv', mode="w")

    return stockData

data = gatherStockDataArima()
data = data["TSLA"]
