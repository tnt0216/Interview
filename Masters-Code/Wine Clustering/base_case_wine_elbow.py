import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_embeddings():
    """
    Loading the wine dataset in pandas dataframe
    :return: scaled data
    """
    # loading wine dataset
    wine_raw = pd.read_csv("wine-clustering.csv")

    # checking data shape
    row, col = wine_raw.shape
    print(f'There are {row} rows and {col} columns')
    print(wine_raw.head(10))

    # to work on copy of the data
    wine_raw_scaled = wine_raw.copy()

    # Scaling the data to keep the different attributes in same range.
    wine_raw_scaled[wine_raw_scaled.columns] = StandardScaler().fit_transform(wine_raw_scaled)
    print(wine_raw_scaled.describe())

    return wine_raw_scaled


def elbow_method(data_scaled):
    # Determining the best value for K be training KMeans model
    # inertia is the sum of the squared distances of samples to their closest cluster center
    inertias = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data_scaled)
        # This tells us how far away the points within a cluster are (small inertias are aimed for)
        # The range of inertias' value starts from zero and goes up
        inertias.append(kmeans.inertia_)

    plt.plot(range(1, 11), inertias, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()


def main():

    print("1. Loading Wine dataset\n")
    data_scaled = load_embeddings()

    elbow_method(data_scaled)

