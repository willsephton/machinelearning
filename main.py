import pandas as pd
import yfinance as yf
from sklearn.preprocessing import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import seaborn as sns

# ! Data Retrival

with open("dataset/list_of_tickers.txt", "r") as file:
    tickers = file.read().splitlines()
    

def gatherStockDataPCAandKMeans():
    stockData = yf.download(tickers, period='1y', interval='1d', group_by='tickers') # Downloads the nasdaq stock data
    closeData = stockData.xs('Close', level=1, axis=1)

    closeData = closeData.T

    removedRows = closeData.dropna(axis=1) #Cleans rows with empty cells

    return removedRows

def gatherStockDataCorrelation():
    stockData = yf.download(tickers, period='1y', interval='1d', group_by='tickers') # Downloads the nasdaq stock data
    closeData = stockData.xs('Close', level=1, axis=1)

    removedRows = closeData.dropna(axis=1) #Cleans rows with empty cells

    return removedRows

# ! PCA
def pcaFunction(PCAandKmeansData):
    pca = PCA(n_components=10)
    reduced_data = pca.fit_transform(PCAandKmeansData)
    return reduced_data

# ! K means Clustering

def kmeansFunction(reduced_data):
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(reduced_data)

    # Obtain the cluster labels for each data point
    cluster_labels = kmeans.labels_

    reduced_data = pd.DataFrame(reduced_data, index=stockData.index[:])

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




# ! Calling the functions

chosen_tickers = ["META", "AVGO", "BKNG", "TSLA"]


# ? PCA and Kmeans Calls
#PCAandKmeansData = gatherStockDataPCAandKMeans()
#kmeansData = pcaFunction(PCAandKmeansData)
#kmeansFunction(kmeansData)

# ? Correlation Calls
correlationData = gatherStockDataCorrelation()
correlationOutput(chosen_tickers, correlationData)


