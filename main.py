import pandas as pd
import yfinance as yf
from sklearn.preprocessing import *
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ! Data Retrival

with open("dataset/list_of_tickers.txt", "r") as file:
    tickers = file.read().splitlines()
    

def gatherStockDataPCAandKMeans():
    stockData = yf.download(tickers, period='1y', interval='1d', group_by='tickers') # Downloads the nasdaq stock data
    closeData = stockData.xs('Close', level=1, axis=1)

    closeData = closeData.T

    removedRows = closeData.dropna(axis=1) #Cleans rows with empty cells

    return removedRows

stockData = gatherStockDataPCAandKMeans()


# ! PCA

pca = PCA(n_components=10)
reduced_data = pca.fit_transform(stockData)

# ! K means Clustering

kmeans = KMeans(n_clusters=4)
kmeans.fit(reduced_data)

# Obtain the cluster labels for each data point
cluster_labels = kmeans.labels_

reduced_data = pd.DataFrame(reduced_data, index=stockData.index[:])

# Add the 'Cluster_Label' column to your DataFrame
reduced_data['Cluster_Label'] = cluster_labels

# Save the updated DataFrame to a new CSV file
reduced_data.to_csv('data_with_clusters.csv')



