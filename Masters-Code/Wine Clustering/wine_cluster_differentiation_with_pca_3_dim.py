# Importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

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
    wine_raw = pd.read_csv("wine_types.csv", header=None)
    wine_raw_scaled = wine_raw.drop(columns=[0])
    wine_types = wine_raw[0].values.astype(str)

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

    pca_3 = PCA(n_components=3)
    pca_3_result = pca_3.fit_transform(df_scaled)

    return pca_3_result


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


def main():
    print("1. Loading Wine dataset\n")
    data_scaled, wine_types = load_embeddings()

    print("\n\n2. Reducing via PCA\n")
    pca_result = pca_embeddings(data_scaled)

    print("\n\n3. HyperTuning the Parameter for KMeans\n")
    optimum_num_clusters = kmean_hyper_param_tuning(data_scaled)
    print("optimum num of clusters =", optimum_num_clusters)

    # Fitting KMeans
    kmeans = KMeans(n_clusters=optimum_num_clusters)
    kmeans.fit(data_scaled)

    # Transforming cluster centers to PCA
    centroids = kmeans.cluster_centers_
    centroids_pca = pca_embeddings(centroids)

    # using plotly express package to plot in 3 dimensions
    df_result = pd.DataFrame(data=pca_result, columns=['PCA 1', 'PCA 2', 'PCA 3'])
    df_result['Cluster'] = kmeans.labels_
    df_result['Wine Type'] = wine_types

    fig = px.scatter_3d(df_result, x='PCA 1', y='PCA 2', z='PCA 3', color='Wine Type', symbol='Cluster',
                        opacity=0.7, size_max=10)

    fig.update_layout(title='Wine Clusters in 3D', scene=dict(xaxis_title='PCA 1', yaxis_title='PCA 2',
                                                              zaxis_title='PCA 3'))
    fig.show()


if __name__ == "__main__":
    main()
