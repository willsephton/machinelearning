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
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from pmdarima import auto_arima

import streamlit as st


# ! Data Retrival

with open("dataset/list_of_tickers.txt", "r") as file:
    tickers = file.read().splitlines()
    

def gatherStockDataPCAandKMeans():
    stockData = yf.download(tickers, period='1y', interval='1d', group_by='tickers') # Downloads the nasdaq stock data
    closeData = stockData.xs('Close', level=1, axis=1)

    closeData = closeData.T

    removedRows = closeData.dropna(axis=1) #Cleans rows with empty cells

    return removedRows

def gatherStockDataCorrelationEDA():
    stockData = yf.download(tickers, period='1y', interval='1d', group_by='tickers') # Downloads the nasdaq stock data
    closeData = stockData.xs('Close', level=1, axis=1)

    removedRows = closeData.dropna(axis=1) #Cleans rows with empty cells

    return removedRows

def gatherStockDataForProphet():
    stockData = yf.download(tickers, period='1y', interval='1d')['Close']
    return stockData


# ! PCA
def pcaFunction(PCAandKmeansData):
    pca = PCA(n_components=10)
    reduced_data = pca.fit_transform(PCAandKmeansData)
    return reduced_data

# ! K means Clustering

def kmeansFunction(reduced_data, PCAandKmeansData):
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(reduced_data)

    # Obtain the cluster labels for each data point
    cluster_labels = kmeans.labels_

    reduced_data = pd.DataFrame(reduced_data, index=PCAandKmeansData.index[:])

    # Add the 'Cluster_Label' column to your DataFrame
    reduced_data['Cluster_Label'] = cluster_labels

    # Save the updated DataFrame to a new CSV file
    st.write(reduced_data)
    # ? Uncomment this later: reduced_data.to_csv('data_with_clusters.csv')

# ! Colleration

def topTenCorrelation(stockData, chosen_ticker):
        st.write(f"Top Ten Correlation for {chosen_ticker}")
        correlated = stockData.corrwith(stockData[chosen_ticker])

        highest11 = correlated.nlargest(11)
        highest10 = highest11.iloc[1:]  # Using iloc to select rows
        st.write(highest10)
        return highest10
    
def bottomTenCorrelation(stockData, chosen_ticker):
        st.write(f"Bottom Ten Correlation for {chosen_ticker}")
        correlated = stockData.corrwith(stockData[chosen_ticker])
        lowest10 = correlated.nsmallest(10)
        #lowest10 = lowest116.iloc[1:]  # Using iloc to select rows
        st.write(lowest10)
        return lowest10

def correlationOutput(chosen_ticker, stockData):
    topTenCorrelation(stockData, chosen_ticker)

    bottomTenCorrelation(stockData, chosen_ticker)


# ! EDA
        
# * Temporal Structure
def stockPricesOverTime(stockData, chosen_ticker):
    
    chosen_stock_data = stockData[chosen_ticker]


    plt.plot(chosen_stock_data.index, chosen_stock_data, label=chosen_ticker)

    # Adding labels and title
    plt.xlabel('Date')
    plt.ylabel(chosen_ticker+' Stock Price')
    plt.title(chosen_ticker+' Stock Prices Over Time')
    plt.legend()  # Show legend with ticker labels
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

    st.pyplot(plt)  # Display the plot


# * Visualize the distribution of observation

def distributionOfStockPricesBox(stockData, chosen_ticker):

    stockData = stockData[chosen_ticker]

    plt.figure(figsize=(8, 6))

    plt.boxplot(stockData, vert=False)
    plt.title('Box Plot - Distribution of Stock Prices')
    plt.xlabel('Stock Prices')
    plt.yticks([])
    plt.grid(axis='x')

    st.pyplot(plt)


def distributionOfStockPricesHistogram(stockData, chosen_ticker):

    stockData = stockData[chosen_ticker]

    plt.figure(figsize=(8, 6))


    # Plotting a histogram for the selected stock's prices
    plt.hist(stockData, bins=30, color='skyblue')
    plt.title(f'Histogram of {chosen_ticker} Stock Prices')
    plt.xlabel('Stock Prices')
    plt.ylabel('Frequency')

    st.pyplot(plt)


# * Investigate the change in distribution over intervals

def monthlyStockLine(stockData, chosen_ticker):
    
    stockData = stockData[chosen_ticker]
    stockData.index = pd.to_datetime(stockData.index)

    # Resample the data into monthly intervals and use mean
    stockDataMonthly = stockData.resample('M').mean()  



    plt.plot(stockDataMonthly.index, stockDataMonthly, label=chosen_ticker)

    # Adding labels and title
    plt.xlabel('Date')
    plt.ylabel(f'{chosen_ticker} Stock Price')
    plt.title(f'{chosen_ticker} Stock Prices Over Time')
    plt.legend()  # Show legend with ticker labels
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

    st.pyplot(plt)

def monthlyStockHistogram(stockData, chosen_ticker):
    
    # Extracting the chosen ticker's data
    stockData = stockData[chosen_ticker]
    
    # Converting the index to a DateTimeIndex if it's not already in that format
    stockData.index = pd.to_datetime(stockData.index)

    # Resampling the data into monthly intervals and use mean
    stockDataMonthly = stockData.resample('M').mean()  

    plt.figure(figsize=(10, 6))

    # Plotting a histogram for the selected stock's monthly average prices
    plt.hist(stockDataMonthly, bins=30, color='skyblue')
    plt.title(f'Histogram of {chosen_ticker} Stock Monthly Average Prices')
    plt.xlabel('Monthly Average Stock Prices')
    plt.ylabel('Frequency')

    st.pyplot(plt)

def weeklyStockBox(stockData, chosen_ticker):

    stockData = stockData[chosen_ticker]
    weekly_data = stockData.resample('W').mean()

    # Plotting the box plot for weekly data
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=weekly_data.values)
    plt.xlabel('Weeks')
    plt.ylabel('Closing Price')
    plt.title(f'Distribution of Weekly Closing Prices for {chosen_ticker}')
    st.pyplot(plt)


# ! Prophet Prediction
    
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

# ! Prophet Intervals 
    
def prophetIntervalWeek(chosen_ticker, prophetData):
    days = 7
    prophetFunction(chosen_ticker, days, prophetData)
def prophetIntervalTwoWeek(chosen_ticker, prophetData):
    days = 14
    prophetFunction(chosen_ticker, days, prophetData)
def prophetIntervalMonth(chosen_ticker, prophetData):
    days = 30
    prophetFunction(chosen_ticker, days, prophetData)


# ! Long Short-Term Memory
    
def createDatasetforLSTM(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def lstm_stocks(chosen_stock, stockData):
    keras = tf.keras
    Sequential = keras.models.Sequential
    Dense = keras.layers.Dense
    LSTM = keras.layers.LSTM
    dataFrame = stockData.reset_index()[chosen_stock]

    scaler = MinMaxScaler()
    dataFrame = scaler.fit_transform(np.array(dataFrame).reshape(-1, 1))
    train_size = int(len(dataFrame) * 0.65)
    test_size = len(dataFrame) - train_size
    train_data, test_data = dataFrame[0:train_size, :], dataFrame[train_size:len(dataFrame), :1]

    time_step = 10
    X_train, Y_train = createDatasetforLSTM(train_data, time_step)
    X_test, Y_test = createDatasetforLSTM(test_data, time_step)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(
        X_train, Y_train,
        epochs=30,
        batch_size=16,
        validation_split=0.1,
        verbose=1,
        shuffle=False
    )

    y_pred = model.predict(X_test)

    plt.plot(Y_test, marker='.', label="true")
    plt.plot(y_pred, 'r', label="prediction")
    plt.ylabel('Value')
    plt.xlabel('Time Step')
    plt.legend()
    st.pyplot(plt)

    fig_prediction = plt.figure(figsize=(10, 8))
    plt.plot(np.arange(0, len(Y_train)), Y_train, 'g', label="history")
    plt.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)), Y_test, marker='.', label="true")
    plt.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)), y_pred, 'r', label="prediction")
    plt.ylabel('Value')
    plt.xlabel('Time Step')
    plt.legend()
    st.pyplot(plt)

# ! Linear Regression

def linearRegressionFunction(listOfDates, chosenStockData, chosen_ticker, linearRegData):
    # Convert input data into numpy arrays and reshape them
    listOfDates = np.asanyarray(listOfDates)  # Convert list of dates to a NumPy array
    chosenStockData = np.asanyarray(chosenStockData)  # Convert stock data to a NumPy array
    listOfDates = np.reshape(listOfDates, (len(listOfDates), 1))  # Reshape date array into (length, 1)
    chosenStockData = np.reshape(chosenStockData, (len(chosenStockData), 1))  # Reshape stock data array into (length, 1)

    # Attempt to load the previously saved model to evaluate its performance
    try:
        pickle_in = open("prediction.pickle", "rb")
        reg = pickle.load(pickle_in)
        xtrain, xtest, ytrain, ytest = train_test_split(listOfDates, chosenStockData, test_size=1)
        best = reg.score(ytrain, ytest)  # Evaluate model accuracy using test data
    except:
        pass  # If loading the model fails, proceed without errors

    # Initialize the variable to hold the best accuracy achieved
    best = 0

    # Train the model iteratively multiple times to find the best accuracy
    for z in range(100):
        xtrain, xtest, ytrain, ytest = train_test_split(listOfDates, chosenStockData, test_size=0.80)
        reg = LinearRegression().fit(xtrain, ytrain)  # Fit a Linear Regression model
        accuracy = reg.score(xtest, ytest)  # Calculate the accuracy of the model
        # Check if the current accuracy is better than the previous best accuracy
        if accuracy > best:
            best = accuracy  
            with open('prediction.pickle', 'wb') as f:
                pickle.dump(reg, f)  # Save the best model using pickle
            print(accuracy)  # Print the current best accuracy

    # Load the best model obtained during training
    pickle_in = open("prediction.pickle", "rb")
    reg = pickle.load(pickle_in)

    # Evaluate the average accuracy of the best model over multiple iterations
    mean = 0
    for i in range(10):
        msk = np.random.rand(len(linearRegData)) < 0.8
        xtest = listOfDates[~msk]
        ytest = chosenStockData[~msk]
        mean += reg.score(xtest, ytest)  # Calculate accuracy using test data

    print("Average Accuracy:", mean / 10)

    # Plot the actual and predicted stock prices
    plt.plot(xtest, ytest, color='blue', linewidth=1, label='Actual Stock Price') 
    plt.plot(xtest, reg.predict(xtest), color='red', linewidth=3, label='Predicted Stock Price')  
    plt.title(f"Linear Regression for {chosen_ticker}") 
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    st.pyplot(plt) 

# ! ARIMA
    
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
    st.pyplot(plt)

# ! Buy/Sell Signals

def createSignals(forecast):
    buy_threshold = 0.1 # Buy threshold is +10%
    sell_threshold = -0.1  # Sell threshold is -10%
    
    buy_count = sum(forecast['yhat'].pct_change() > buy_threshold)
    sell_count = sum(forecast['yhat'].pct_change() < sell_threshold)
    
    if buy_count > sell_count:
        return 'Buy'
    else:
        return 'Sell'

# ! StreamLit Stuff


def main():
    st.title('Machine Learning Assessment Application')

    chosen_tickers = ["META", "AVGO", "BKNG", "TSLA"]
    chosen_ticker = st.sidebar.selectbox('Select a Ticker', chosen_tickers)

    option = st.sidebar.selectbox(
        'Select an Option',
        ('Home', 'PCA and Kmeans clustering', 'Correlation', 'EDA', 'Prophet Forecast', 'ARIMA Forecast', 'LSTM Forecast', 'Linear Regression Forecast')
    )

    if option == "Home":
        st.subheader('COM624 - Will Sephton')
        st.write("My chosen stocks from the 4 clusters are:")
        st.write("Meta Platforms Inc - META")
        st.write("Broadcom Inc - AVGO")
        st.write("Booking Holdings Inc - BKNG")
        st.write("Tesla Inc - TSLA")
        st.write("In order to choose these stocks I created a random number generator in Python and assigned each stock in a cluster with a number.")
        st.write("All data is loaded in real time and downloaded whenever there is an option change or ticker change.")

    
    if option == 'PCA and Kmeans clustering':
        st.subheader('Dataset after PCA and Kmeans clustering')
        st.write("The cluster label is listed on the far right of the dataset")
        PCAandKmeansData = gatherStockDataPCAandKMeans()
        kmeansData = pcaFunction(PCAandKmeansData)
        kmeansFunction(kmeansData, PCAandKmeansData)

    if option == 'Correlation':
        st.subheader('Top Ten and Bottom Ten Correlated Stocks')
        correlationData = gatherStockDataCorrelationEDA()
        correlationOutput(chosen_ticker, correlationData)

    if option == 'EDA':
        st.subheader('EDA')
        edaData = gatherStockDataCorrelationEDA()
        st.write("Temporal Structure")
        st.write(f"Stock Prices Over Time for {chosen_ticker}")
        stockPricesOverTime(edaData, chosen_ticker)
        st.write("Visualize the distribution of observation")
        st.write(f"Distribution of Stock Prices for {chosen_ticker} (Box Chart)")
        distributionOfStockPricesBox(edaData, chosen_ticker)
        st.write(f"Distribution of Stock Prices for {chosen_ticker} (Histogram)")
        distributionOfStockPricesHistogram(edaData, chosen_ticker)
        st.write("Investigate the change in distribution over intervals")
        st.write(f"Monthly Stock Prices for {chosen_ticker}")
        monthlyStockHistogram(edaData, chosen_ticker)
        st.write(f"Weekly Stock Prices for {chosen_ticker}")
        weeklyStockBox(edaData, chosen_ticker)

    if option == 'Prophet Forecast':
        st.subheader('Prophet Forecast')
        prophetData = gatherStockDataForProphet()
        days = st.slider('Select Forecasting Period (in days)', min_value=1, max_value=365, value=30)
        prophetFunction(chosen_ticker, days, prophetData)

    if option == 'ARIMA Forecast':
        st.subheader('Arima Forecast')
        arimaData = gatherStockDataForProphet()
        specficStock = arimaData[chosen_ticker]
        arimaFunction(specficStock, chosen_ticker)
    
    if option == 'LSTM Forecast':
        st.subheader('LSTM Forecast')
        lstmData = gatherStockDataForProphet()
        lstm_stocks(chosen_ticker, lstmData)

    if option == 'Linear Regression Forecast':
        st.subheader('Linear Regression Forecast')
        linearRegData = gatherStockDataForProphet()
        listOfDates = list(range(0, int(len(linearRegData))))
        chosenStockData = linearRegData[chosen_ticker]
        linearRegressionFunction(listOfDates, chosenStockData, chosen_ticker, linearRegData)

if __name__ == "__main__":
    main()