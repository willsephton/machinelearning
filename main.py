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
    # ? Uncomment this later: reduced_data.to_csv('data_with_clusters.csv')

# ! Colleration

def topTenCorrelation(stockData, chosen_tickers, x):
        print(f"Top Ten Correlation for {chosen_tickers[x]}")
        correlated = stockData.corrwith(stockData[chosen_tickers[x]])

        highest11 = correlated.nlargest(11)
        highest10 = highest11.iloc[1:]  # Using iloc to select rows
        print(highest10)
        return highest10
    
def bottomTenCorrelation(stockData, chosen_tickers, x):
        print(f"Bottom Ten Correlation for {chosen_tickers[x]}")
        correlated = stockData.corrwith(stockData[chosen_tickers[x]])
        lowest10 = correlated.nsmallest(10)
        #lowest10 = lowest116.iloc[1:]  # Using iloc to select rows
        print(lowest10)
        return lowest10

def correlationOutput(chosen_tickers, stockData):
    for x in range(len(chosen_tickers)):

        topTenCorrelation(stockData, chosen_tickers, x)

        bottomTenCorrelation(stockData, chosen_tickers, x)


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

    plt.show()  # Display the plot


# * Visualize the distribution of observation

def distributionOfStockPricesBox(stockData, chosen_ticker):

    stockData = stockData[chosen_ticker]

    plt.figure(figsize=(8, 6))

    plt.boxplot(stockData, vert=False)
    plt.title('Box Plot - Distribution of Stock Prices')
    plt.xlabel('Stock Prices')
    plt.yticks([])
    plt.grid(axis='x')

    plt.show()

def distributionOfStockPricesHistogram(stockData, chosen_ticker):

    stockData = stockData[chosen_ticker]

    plt.figure(figsize=(8, 6))


    # Plotting a histogram for the selected stock's prices
    plt.hist(stockData, bins=30, color='skyblue')
    plt.title(f'Histogram of {chosen_ticker} Stock Prices')
    plt.xlabel('Stock Prices')
    plt.ylabel('Frequency')

    plt.show()


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

    plt.show()  # Display the plot

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

    plt.show()

def weeklyStockBox(stockData, chosen_ticker):

    stockData = stockData[chosen_ticker]
    weekly_data = stockData.resample('W').mean()

    # Plotting the box plot for weekly data
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=weekly_data.values)
    plt.xlabel('Weeks')
    plt.ylabel('Closing Price')
    plt.title(f'Distribution of Weekly Closing Prices for {chosen_ticker}')
    plt.show()


# ! Prophet Prediction
    
def prophetFunction(chosen_ticker, days):
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
    future = model.make_future_dataframe(periods=days)  # forecasting for 1 year from now.
    forecast = model.predict(future)

    figure = model.plot(forecast)
    plt.title(f"Facebook Prediction for {chosen_ticker}")
    plt.show()

# ! Prophet Intervals 
    
def prophetIntervalWeek(chosen_ticker):
    days = 7
    prophetFunction(chosen_ticker, days)
def prophetIntervalTwoWeek(chosen_ticker):
    days = 14
    prophetFunction(chosen_ticker, days)
def prophetIntervalMonth(chosen_ticker):
    days = 30
    prophetFunction(chosen_ticker, days)


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
    plt.show()

    fig_prediction = plt.figure(figsize=(10, 8))
    plt.plot(np.arange(0, len(Y_train)), Y_train, 'g', label="history")
    plt.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)), Y_test, marker='.', label="true")
    plt.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)), y_pred, 'r', label="prediction")
    plt.ylabel('Value')
    plt.xlabel('Time Step')
    plt.legend()
    plt.show()

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
    plt.show()  
# ! Calling the functions

chosen_tickers = ["META", "AVGO", "BKNG", "TSLA"]
chosen_ticker = input("META, AVGO, BKNG, or TSLA:   ").upper()


# ? PCA and Kmeans Calls
#PCAandKmeansData = gatherStockDataPCAandKMeans()
#kmeansData = pcaFunction(PCAandKmeansData)
#kmeansFunction(kmeansData, PCAandKmeansData)

# ? Correlation Calls
#correlationData = gatherStockDataCorrelationEDA()
#correlationOutput(chosen_tickers, correlationData)

# ? EDA Calls

#edaData = gatherStockDataCorrelationEDA()
#stockPricesOverTime(edaData, chosen_ticker)
#distributionOfStockPricesBox(edaData, chosen_ticker)
#distributionOfStockPricesHistogram(edaData, chosen_ticker)
#monthlyStockHistogram(edaData, chosen_ticker)
#weeklyStockBox(edaData, chosen_ticker)

# ? Forecasting Calls

#prophetData = gatherStockDataForProphet()
#prophetIntervalWeek(chosen_ticker)
#prophetIntervalTwoWeek(chosen_ticker)
#prophetIntervalMonth(chosen_ticker)

#lstmData = gatherStockDataForProphet()
#lstm_stocks(chosen_ticker, lstmData)

linearRegData = gatherStockDataForProphet()
listOfDates = list(range(0, int(len(linearRegData))))
chosenStockData = linearRegData[chosen_ticker]
linearRegressionFunction(listOfDates, chosenStockData, chosen_ticker, linearRegData)