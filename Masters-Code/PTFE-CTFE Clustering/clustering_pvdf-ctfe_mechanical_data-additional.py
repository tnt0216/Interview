# Importing packages
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
    raw_data = pd.read_csv("PVDF-CTFE Mechanical Data_DW-Additions.csv", dtype={'Lot': str})

    # Only want data pertaining to procedures M-D and also have a head speed of 500 so filtering data
    filtered_data = raw_data[(raw_data['Procedure'].str.contains("M-D")) & (raw_data['Head Speed'] == 500)]

    # Extracting the "sample names" column since cannot perform StandardScalar transformation on string
    sample_names = filtered_data["Lot"].values

    # raw_data_scaled = raw_data.drop(columns=["Lot"])

    # Pulling the data from the csv file that we want as dimensions
    selected_columns = list(raw_data.columns[1:9])
    raw_data_scaled = filtered_data[selected_columns]

    # Scaling the float data to keep the different attributes in same range.
    # LOOK INTO HOW STANDARDSCALAR WORKS!!!
    raw_data_scaled[raw_data_scaled.columns] = StandardScaler().fit_transform(raw_data_scaled)
    print(raw_data_scaled.describe())

    return raw_data_scaled, sample_names


def pca_embeddings(df_scaled):

    # making the dimensions 3 so that it is easier to visualize
    pca_3 = PCA(n_components=3)
    pca_3_result = pca_3.fit_transform(df_scaled)

    return pca_3_result


def kmean_hyper_param_tuning(data):
    """
    Hyper parameter tuning to select the best from all the parameters on the basis of silhouette_score.

    :param data: dimensionality reduced data after applying PCA
    :return: best number of clusters for the model (used for KMeans n_clusters)
    """

    # candidate values for our number of cluster ***NOTE: this will need to change depending on the dataset size. If you
    # have a small dataset it would not make since to use 40 clusters per se.
    parameters = [2, 4, 6, 8, 10]

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
    raw_data_scaled, sample_names = load_embeddings()

    print("\n\n2. Reducing via PCA\n")
    pca_result = pca_embeddings(raw_data_scaled)

    print("\n\n3. HyperTuning the Parameter for KMeans\n")
    optimum_num_clusters = kmean_hyper_param_tuning(raw_data_scaled)
    print("optimum num of clusters =", optimum_num_clusters)

    # Fitting KMeans
    kmeans = KMeans(n_clusters=optimum_num_clusters)
    kmeans.fit(raw_data_scaled)

    # Transforming cluster centers to PCA
    centroids = kmeans.cluster_centers_
    centroids_pca = pca_embeddings(centroids)

    # using plotly express package to plot in 3 dimensions
    df_result = pd.DataFrame(data=pca_result, columns=['PCA 1', 'PCA 2', 'PCA 3'])
    df_result['Cluster'] = kmeans.labels_
    df_result['sample names'] = sample_names

    fig = px.scatter_3d(df_result, x='PCA 1', y='PCA 2', z='PCA 3', color='sample names', symbol='Cluster',
                        opacity=0.7, size_max=10)

    fig.update_layout(title='Clusters in 3D', scene=dict(xaxis_title='PCA 1', yaxis_title='PCA 2', zaxis_title='PCA 3'))

    fig.show()


if __name__ == "__main__":
    main()
