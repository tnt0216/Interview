# Source found from: https://medium.com/swlh/k-means-clustering-on-high-dimensional-data-d2151e1a4240
# Github code from: https://github.com/Shivangi0503/Wine_Clustering_KMeans/blob/main/data/wine-clustering.csv
# This code runs with 13 dimensions which will be similar to what we plan to do with our data. We use PCA to reduce
# the dimensions to allow us to us clustering techniques.
# Additional research and information can be found here: https://towardsdatascience.com/kmeans-hyper-parameters-
# explained-with-examples-c93505820cd3

# Importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
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
    wine_raw = pd.read_csv("wine-clustering-wine-types.csv", dtype={'Wine Type': str})

    # Extracting the "Wine Type" column since cannot perform StandardScalar transformation on string
    wine_types = wine_raw["Wine Type"].values
    wine_raw_scaled = wine_raw.drop(columns=["Wine Type"])

    # Scaling the float data to keep the different attributes in same range.
    # LOOK INTO HOW STANDARDSCALAR WORKS!!!
    wine_raw_scaled[wine_raw_scaled.columns] = StandardScaler().fit_transform(wine_raw_scaled)
    print(wine_raw_scaled.describe())

    return wine_raw_scaled, wine_types


def pca_embeddings(df_scaled):
    """To reduce the dimensions of the wine dataset we use Principal Component Analysis (PCA).
    Here we reduce it from 13 dimensions to 2.

    :param df_scaled: scaled data
    :return: pca result, pca for plotting graph
    """

    pca_2 = PCA(n_components=2)
    pca_2_result = pca_2.fit_transform(df_scaled)
    print('Explained variation per principal component: {}'.format(pca_2.explained_variance_ratio_))
    print('Cumulative variance explained by 2 principal components: {:.2%}'.format(
        np.sum(pca_2.explained_variance_ratio_)))

    # Results from pca.components_
    dataset_pca = pd.DataFrame(abs(pca_2.components_), columns=df_scaled.columns, index=['PC_1', 'PC_2'])
    print('\n\n', dataset_pca)

    print("\n*************** Most important features *************************")
    print('As per PC 1:\n', (dataset_pca[dataset_pca > 0.3].iloc[0]).dropna())
    print('\n\nAs per PC 2:\n', (dataset_pca[dataset_pca > 0.3].iloc[1]).dropna())
    print("\n******************************************************************")

    return pca_2_result, pca_2


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


def visualizing_results(pca_result, label, centroids_pca, wine_types):
    """ Visualizing the clusters

    :param wine_types: The first column of the datasheet that contains the type of wine (i.e. red, white, pink) is an
                        array
    :param pca_result: PCA applied data
    :param label: K Means labels
    :param centroids_pca: PCA format K Means centroids
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
        'Red': 'red',
        'White': 'black',
        'Pink': 'pink'
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
        cluster_data = pca_result[label == cluster_num]
        # Plotting the data points with different symbols for each cluster
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], marker=markers[cluster_num], alpha=0.5, s=100,
                    label=f'Cluster {cluster_num+ 1 }')

    # Plotting the centroids
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, linewidths=1.5,
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
            cluster_wine_data = pca_result[(label == cluster_num) & (wine_types == wine_type)]
            # Plotting the data points
            plt.scatter(cluster_wine_data[:, 0], cluster_wine_data[:, 1], marker=markers[cluster_num], alpha=0.7, s=100,
                        c=wine_colors[wine_type], label=f'Cluster {cluster_num+1}, {wine_type}')

        # Plotting the centroids for each cluster
        cluster_centroids_pca = centroids_pca[cluster_num].reshape(1, -1)
        plt.scatter(cluster_centroids_pca[:, 0], cluster_centroids_pca[:, 1], marker='X', s=200, linewidths=1.5,
                    color='blue', edgecolors="black", lw=1.5, label=f'Centroid, Cluster {cluster_num+1}')

        plt.title('Wine Clusters & Wine Types With PCA')
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')

    # Adding the legend outside the subplots
    # The 'bbox_to_anchor' parameter specifies the position of the legend & 'loc' specifies the location of the legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Adjusts spacing between subplots to avoid overlapping
    plt.tight_layout()
    plt.show()


def main():
    print("1. Loading Wine dataset\n")
    data_scaled, wine_types = load_embeddings()

    print("\n\n2. Reducing via PCA\n")
    pca_result, pca_2 = pca_embeddings(data_scaled)

    print("\n\n3. HyperTuning the Parameter for KMeans\n")
    optimum_num_clusters = kmean_hyper_param_tuning(data_scaled)
    print("optimum num of clusters =", optimum_num_clusters)

    # Fitting KMeans
    kmeans = KMeans(n_clusters=optimum_num_clusters)
    kmeans.fit(data_scaled)
    centroids = kmeans.cluster_centers_
    centroids_pca = pca_2.transform(centroids)

    print("\n\n4. Visualizing the data")
    visualizing_results(pca_result, kmeans.labels_, centroids_pca, wine_types)


if __name__ == "__main__":
    main()
