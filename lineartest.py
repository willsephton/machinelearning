import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import yfinance as yf

def linear_regression(dates, prices, chosen_stock):
    # Convert to numpy array and reshape them
    dates = np.asanyarray(dates)
    prices = np.asanyarray(prices)
    dates = np.reshape(dates, (len(dates), 1))
    prices = np.reshape(prices, (len(prices), 1))

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(dates, prices, test_size=0.2, random_state=42)

    # Train the linear regression model
    reg = LinearRegression().fit(x_train, y_train)

    # Model evaluation
    accuracy = reg.score(x_test, y_test)
    print(f"Accuracy of Linear Regression Model: {accuracy}")

    # Plot Predicted vs Actual Data
    plt.plot(x_test, y_test, color='green', linewidth=1, label='Actual Price')
    plt.plot(x_test, reg.predict(x_test), color='blue', linewidth=3, label='Predicted Price')
    plt.title(f"Linear Regression {chosen_stock} | Time vs. Price ")
    plt.legend()
    plt.xlabel('Date Integer')
    plt.ylabel('Stock Price')
    plt.show()




with open("dataset/list_of_tickers.txt", "r") as file:
    tickers = file.read().splitlines()

def gatherStockDataForProphet():
    stockData = yf.download(tickers, period='1y', interval='1d')['Close']
    return stockData

# Assuming 'nasdaq_data_original.csv' contains the necessary data
original_nasdaq_data = gatherStockDataForProphet()

# Extract dates, prices, and chosen stock data
dates = list(range(0, len(original_nasdaq_data)))
chosen_stock = "TSLA"  # Change this to the desired stock symbol
prices = original_nasdaq_data[chosen_stock]

# Perform linear regression
linear_regression(dates, prices, chosen_stock)
