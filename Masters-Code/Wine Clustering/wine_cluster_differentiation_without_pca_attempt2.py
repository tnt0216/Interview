# Importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans


def load_embeddings():
    """
    Loading the wine dataset in pandas dataframe
    :return: scaled data
    """
    # loading wine dataset
    # wine_raw = pd.read_csv("wine-clustering.csv") # for data without wine types column
    wine_raw = pd.read_csv("wine_types.csv", header=None)
    wine_raw_scaled = wine_raw.drop(columns=[0])
    wine_types = wine_raw[0].values.astype(str)

    # Scaling the float data to keep the different attributes in same range.
    # LOOK INTO HOW STANDARDSCALAR WORKS!!!
    wine_raw_scaled[wine_raw_scaled.columns] = StandardScaler().fit_transform(wine_raw_scaled)
    print(wine_raw_scaled.describe())

    return wine_raw_scaled, wine_types


def kmean_hyper_param_tuning(data):
    """
    Hyper parameter tuning to select the best from all the parameters on the basis of silhouette_score.

    :param data: dimensionality reduced data after applying PCA
    :return: best number of clusters for the model (used for KMeans n_clusters)
    """
    # candidate values for our number of cluster
    parameters = [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40]

    # instantiating ParameterGrid, pass number of clusters as input
    parameter_grid = ParameterGrid({'n_clusters': parameters})

    best_score = -1
    kmeans_model = KMeans()  # instantiating KMeans model
    silhouette_scores = []

    # evaluation based on silhouette_score (Look into this)
    for p in parameter_grid:
        kmeans_model.set_params(**p)  # set current hyper parameter
        kmeans_model.fit(data)  # fit model on wine dataset, this will find clusters based on parameter p

        ss = metrics.silhouette_score(data, kmeans_model.labels_)  # calculate silhouette_score
        silhouette_scores += [ss]  # store all the scores

        print('Parameter:', p, 'Score', ss)

        # check p which has the best score
        if ss > best_score:
            best_score = ss
            best_grid = p

    # plotting silhouette score (This is used as a metric to calculate the goodness of a clustering technique)
    plt.bar(range(len(silhouette_scores)), list(silhouette_scores), align='center', color='#722f59', width=0.5)
    plt.xticks(range(len(silhouette_scores)), list(parameters))
    plt.title('Silhouette Score', fontweight='bold')
    plt.xlabel('Number of Clusters')
    plt.show()

    return best_grid['n_clusters']


def visualizing_results(data, label, centroids, wine_types):
    """ Visualizing the clusters

    :param centroids: The centroids without the use of PCA
    :param data: The raw data without PCA
    :param wine_types: The first column of the datasheet that contains the type of wine (i.e. red, white, pink) is an
                        array
    :param label: K Means labels
    """
    # Getting the unique wine types and create a colormap for them
    # This ensures that each type is only considered once
    unique_wine_types = np.unique(wine_types)
    # Determines the number of colors needed
    num_colors = len(unique_wine_types)
    # Creates a colormap. The 'tab10' is a specific colormap from Matplotlib library
    colormap = plt.cm.get_cmap('tab10', num_colors)

    # Creating a dictionary for the different wine colors (red, white, pink, etc.)
    wine_colors = {
        '1': 'red',
        '2': 'black',
        '3': 'pink'
    }

    # Creating a list of colors for the data points. Maps each wine type to its color
    data_point_colors = [wine_colors[wine_type] for wine_type in wine_types]

    # Creating markers for the different clusters
    markers = ['o', '*', '^']

    # ------------------ Using Matplotlib for plotting-----------------------

    # Plotting the clusters with symbols
    plt.figure()

    # Iterating through the clusters
    for cluster_num in range(max(label) + 1):

        # Filtering the data points in 'pca_result' array that belong to current cluster
        cluster_data = data[label == cluster_num]
        # Plotting the data points with different symbols for each cluster
        plt.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], marker=markers[cluster_num], alpha=0.5, s=100,
                    label=f'Cluster {cluster_num+ 1 }')

    # Plotting the centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, linewidths=1.5,
                color='red', edgecolors="black", lw=1.5, label='Centroid')

    plt.title('Wine Clusters')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    plt.show()

    # Plotting with symbols for the clusters and colors for the wine type
    plt.figure()

    for cluster_num in range(max(label) + 1):

        # Plotting the data points of each cluster with different colors
        for i, wine_type in enumerate(np.unique(wine_types)):

            # Establishing relationships for cluster and wine types so can plot shapes for clusters and colors of types
            # of wine
            cluster_wine_data = data[(label == cluster_num) & (wine_types == wine_type)]
            # Plotting the data points
            plt.scatter(cluster_wine_data.iloc[:, 0], cluster_wine_data.iloc[:, 1], marker=markers[cluster_num], alpha=0.7, s=100,
                        c=wine_colors[wine_type], label=f'Cluster {cluster_num+1}, {wine_type}')

        # Plotting the centroids for each cluster
        cluster_centroids = centroids[cluster_num].reshape(1, -1)
        plt.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], marker='X', s=200, linewidths=1.5,
                    color='blue', edgecolors="black", lw=1.5, label=f'Centroid, Cluster {cluster_num+1}')

        plt.title('Wine Clusters & Wine Types Without PCA')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

    # Adding the legend outside the subplots
    # The 'bbox_to_anchor' parameter specifies the position of the legend & 'loc' specifies the location of the legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Adjusts spacing between subplots to avoid overlapping
    plt.tight_layout()
    plt.show()


def main():
    print("1. Loading Wine dataset\n")
    data_scaled, wine_types = load_embeddings()

    print("\n\n3. HyperTuning the Parameter for KMeans\n")
    optimum_num_clusters = kmean_hyper_param_tuning(data_scaled)
    print("optimum num of clusters =", optimum_num_clusters)

    # Fitting KMeans
    kmeans = KMeans(n_clusters=optimum_num_clusters)
    kmeans.fit(data_scaled)
    centroids = kmeans.cluster_centers_

    print("\n\n4. Visualizing the data")
    visualizing_results(data_scaled, kmeans.labels_, centroids, wine_types,)


if __name__ == "__main__":
    main()
