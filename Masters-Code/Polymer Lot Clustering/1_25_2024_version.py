"""Importing packages"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import os
import warnings

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean, cityblock, cosine


def save_plot(fig_or_plt, filename, dataset_name, folder_path=""):
    """
    This function takes in a figure or a plot along with a filename and path to save each of the generated plots to a
    specified location on the computer (specified with folder path).
    """

    """This adds specification to the filename for which dataset the plot belongs to"""
    filename, extension = os.path.splitext(filename)
    if extension:
        filename = f"{filename}_{dataset_name}{extension}"
    else:
        filename = f"{filename}_{dataset_name}"

    """Creates the full path for the save file. If there is a folder path then the folder path and the filename are 
    joined to create the full path if not then the full path is set to the name of the file."""
    if folder_path:
        full_path = os.path.join(folder_path, filename)
    else:
        full_path = filename

    """Saving the figure if not a figure saving the plot"""
    if isinstance(fig_or_plt, plt.Figure):
        fig_or_plt.savefig(full_path, bbox_inches='tight')
    else:
        plt.savefig(full_path, bbox_inches='tight')


def save_tables(table_data, pca_cluster_data, dataset_name, folder_path=""):
    """
    This function creates and saves the clustering result tables to Excel files
    """

    """Creating specified filenames for different tables"""
    pca_components_filename = f'{dataset_name}_PCA_components_table.xlsx'
    pca_cluster_filename = f'{dataset_name}_PCA_cluster_table.xlsx'

    """Printing the clustered data to an Excel file"""
    pca_cluster_data.to_excel(os.path.join(folder_path, pca_cluster_filename), sheet_name='sheet1', index=False)
    print("\nTable with all PCA Component Cluster Assignments:")
    print(pca_cluster_data.to_string(index=False))

    """Creating DataFrame from table_data"""
    table_df = pd.DataFrame(table_data)
    print("\nTable with PCA Components, Optimized Clusters, and Variance Explained:\n")
    print(table_df.to_string(index=False))

    """Printing the tables output to an Excel file with a specific file path"""
    table_df.to_excel(os.path.join(folder_path, pca_components_filename), sheet_name='sheet1', index=False)


def sanitize_filename(filename):
    """
    This function is used to make sure that there are no invalid characters in filenames
    """

    invalid_chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename


def load_embeddings():
    """
    This function takes in an Excel file with multiple sheets and pulls data pertaining to columns D through G and
    inserts the data in a DataFrame. The data from the various sheets in the Excel file are inputted into the DataFrame
    so that each row of the DataFrame pertains to an individual sheet. The data pulled from the columns in a single
    sheet were each made into lists and then concat to make the individual row in the Dataframe. StandardScaler is
    applied to the DataFrame for standardization. Multiple standardizations are implemented in this function when
    slightly modified. The output of this function are 2 DataFrames one containing the frequency measurements (used for
    graphing) and another with all the data pertaining to storage and loss moduli as well as viscosity. Also, this
    outputs the names of all the samples from the Excel file for the clustering analysis.
    """

    """Loading excel file"""
    #raw_data = "2055-2060 Series 120C.xlsx"
    raw_data = "2055-2060 Series 170C.xlsx"

    """Creating an ExcelFile object to read the Excel file"""
    xls = pd.ExcelFile(raw_data)

    """Getting a list of all the sheet names (multiple sheets in the excel file)"""
    sheet_names = xls.sheet_names

    """Creating empty lists to store individual data DataFrames and sample names"""
    sample_names = []
    concatenated_rows = []
    frequencies_rows = []

    """Looping through each sheet and reading data into a dataframe"""
    for index, sheet_name in enumerate(sheet_names):

        # Skip the first sheet
        if index == 0:
            print("Ed's Calculations: ")
            eds_data = pd.read_excel(xls, sheet_name=sheet_name, usecols="C:E")
            print(eds_data)
            continue

        """Reading in the data from the current sheet into a DataFrame we want only values pertaining to columns D
        through G since indexing starts at 0 column D is index 3 and so on"""
        # df = pd.read_excel(xls, sheet_name=sheet_name, usecols=[3, 4, 5, 6])
        df = pd.read_excel(xls, sheet_name=sheet_name, usecols="D:G")

        """Need to take only values from rows 2 through 14 since there are NaN values for the first 7 sheets
        PCA does not support NaN values, so I truncated the data for the last files that had additional data
        (This was implemented for a different dataset, however, it works for this dataset)"""
        df = df.iloc[0:14, :]

        """Placing all the values for frequency, storage modulus (G'), loss modulus (G''), and viscosity in their own
        lists so that they can be added into the DataFrames"""
        frequencies_row = df.iloc[:, 0].tolist()
        g_prime = df.iloc[:, 1].tolist()
        g_double_prime = df.iloc[:, 2].tolist()
        viscosity = df.iloc[:, 3].tolist()

        """HERE ARE THE LOGGED VALUES!"""
        # g_prime = [np.log10(value) for value in g_prime]
        # g_double_prime = [np.log10(value) for value in g_double_prime]
        # viscosity = [np.log10(value) for value in viscosity]

        """Concatenating the standardized values"""
        concatenated_row = g_prime + g_double_prime + viscosity
        frequencies_rows.append(frequencies_row)
        concatenated_rows.append(concatenated_row)

        """Storing the sheet name as the sample name"""
        sample_names.append(sheet_name)

    """Creating a DataFrame with the concatenated rows"""
    concatenated_data = pd.DataFrame(concatenated_rows)
    print("\nRaw Data:")
    print(concatenated_data)
    print("")

    """HERE ARE THE STANDARDSCALERS!"""
    scaler = StandardScaler(with_std=False)
    # scaler = StandardScaler()

    """Applying standardscaler to the dataset"""
    concatenated_data_scaled = scaler.fit_transform(concatenated_data)
    concatenated_data_df = pd.DataFrame(concatenated_data_scaled, columns=concatenated_data.columns)
    print("Data After Standardizing:")
    print(concatenated_data_df)
    print("")

    frequency_data = pd.DataFrame(frequencies_rows)
    # print("Frequency Data:")
    # print(frequency_data)
    # print("")

    return sample_names, concatenated_data_df, frequency_data, eds_data


def pca_embeddings(data, dataset_name, num_components):
    """
    This function takes in the scaled DataFrame (data) and the number of components to iterate over for the data. Inside
    the function the PCA histograms that show how PCA weighted each feature in the dataset. This function also produces
    the cumulative variance plots, so we can see how increasing number of PCs contributes to increased captured variance
    in the dataset. The function then returns the pca results (pca weighted coefficients) and the cumulative variance
    """

    """This is where PCA is actually applied to the dataset"""
    pca = PCA(n_components=num_components)  # Defining how many PCs to use
    pca_result = pca.fit_transform(data)  # Transforming the data to the specified PCs to use

    num_features = data.shape[1]

    """Checking if there's only one component (PC) because then we do not use subplots. If there is more than one 
    component then subplots have to be made so that we can see how each feature in the component (PC) was weighted"""
    if num_components == 1:
        """Creating the histogram"""
        fig, ax = plt.subplots()
        ax.bar(range(num_features), pca.components_[0], tick_label=data.columns)
        ax.set_title("PCA 1 Component Analysis")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Weight")
        plt.xticks(rotation=90)

        """Calling to save function"""
        save_plot(fig, f"PCA 1.png", dataset_name, r"C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running "
                                                   r"Data\Last Run\PCA Analysis")

    else:

        """Creating subplots for PCA component analysis"""
        fig, axes = plt.subplots(num_components, 1, sharex=True)

        """Creating the histograms"""
        for i, ax in enumerate(axes):
            ax.bar(range(num_features), pca.components_[i], tick_label=data.columns)
            ax.set_title(f"PCA {i + 1} Component Analysis")
            ax.set_xlabel("Feature")
            ax.set_ylabel("Weight")
            plt.xticks(rotation=90)

        plt.tight_layout()

        """Only saving and the first 5 PCs since becomes unhelpful with more"""
        if num_components <= 5:
            """Calling to save function"""
            save_plot(fig, f"PCA {i + 1}.png", dataset_name,
                      os.path.join(r"C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running Data\Last Run\PCA "
                                   r"Analysis"))

    """Calculating variance ratios"""
    variance = pca.explained_variance_ratio_
    cumulative_variance = variance.cumsum()

    """Creating the variance percentage plot"""
    fig_percent_variance, ax_percent_variance = plt.subplots(figsize=(8, 4))
    ax_percent_variance.bar(range(1, num_components + 1), variance * 100, alpha=0.5, align='center', label='Variance')
    ax_percent_variance.step(range(1, num_components + 1), cumulative_variance * 100, where='mid',
                             label='Cumulative Variance')
    ax_percent_variance.set_title('Variance Percentage For PCA Components')
    ax_percent_variance.set_xlabel('PCA')
    ax_percent_variance.set_ylabel('Variance (%)')
    ax_percent_variance.legend(loc='best')

    """Only saving and the first 5 PCs since becomes unhelpful with more"""
    if num_components <= 5:
        """Calling to save function"""
        save_plot(fig_percent_variance, f"Percent_Variance_PCA_{num_components}.png", dataset_name,
                  os.path.join(
                      r"C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running Data\Last Run\Percent Variance"))

    return pca_result, cumulative_variance


def kmean_hyper_param_tuning(data, num_components, dataset_name):
    """
    This function takes in the scaled DataFrame (data) after PCA has been applied and the number of components to
    iterate over for the data. Inside of this function the silhouette scores and elbow scores are calculated and
    histograms/plots are created for visual aid. In addition to these plots the best number of clusters for the model
    is returned which is used for KMeans n_clusters.
    """

    """Candidate values for our number of cluster ***NOTE: this will need to change depending on the dataset size. If 
    you have a small dataset it would not make since to use 40 clusters."""
    parameters = [2, 3, 4, 5, 6]

    """Instantiating ParameterGrid, pass number of clusters as input"""
    parameter_grid = ParameterGrid({'n_clusters': parameters})

    """Silhouette Score Things"""
    best_score = -1
    """Instantiating KMeans model"""
    kmeans_model = KMeans()
    silhouette_scores = []

    """Elbow Things"""
    best_inertia = np.inf
    inertias = []

    """Evaluation based on silhouette_score (Look into this)"""
    for p in parameter_grid:

        """Setting current hyper parameter"""
        kmeans_model.set_params(**p)
        """Fitting the model to the dataset to find clusters based on the parameter p"""
        kmeans_model.fit(data)

        """Calculating the silhouette scores and storing them"""
        ss = metrics.silhouette_score(data, kmeans_model.labels_)
        silhouette_scores += [ss]

        inertia = kmeans_model.inertia_
        inertias.append(inertia)

        # print('Parameter:', p, 'Score', ss, 'Inertia:', inertia)

        """Checking p which has the best silhouette score"""
        if ss > best_score:
            best_score = ss
            best_grid = p

        """Checking for the "elbow" in the inertia plot"""
        if inertia < best_inertia:
            best_inertia = inertia
            best_inertia_grid = p

    """Plotting silhouette scores (This is used as a metric to calculate the how well defined and compact the clusters 
    are"""
    fig_silhouette, ax_silhouette = plt.subplots()
    ax_silhouette.bar(range(len(silhouette_scores)), list(silhouette_scores), align='center', color='#722f59',
                      width=0.5)
    ax_silhouette.set_xticks(range(len(silhouette_scores)))
    ax_silhouette.set_xticklabels(list(parameters))
    ax_silhouette.set_title('Silhouette Score', fontweight='bold')
    ax_silhouette.set_xlabel('Number of Clusters')

    """Calling to save function"""
    save_plot(fig_silhouette, f"Silhouette_PCA_{num_components}.png", dataset_name, os.path.join(
        r"C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running Data\Last Run\Silhouette Scores"))

    """Plotting the elbow graph"""
    fig_elbow, ax_elbow = plt.subplots()
    ax_elbow.plot(parameters, inertias, marker='o', linestyle='-')
    ax_elbow.set_title('Elbow Method for Optimal k')
    ax_elbow.set_xlabel('Number of Clusters (k)')
    ax_elbow.set_ylabel('Inertia')
    ax_elbow.set_xticks(parameters)

    """Calling to save function"""
    save_plot(fig_elbow, f"Elbow_PCA_{num_components}.png", dataset_name, os.path.join(
        r"C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running Data\Last Run\Elbow Method"))

    # print("")
    # print("Best number of clusters based on Silhouette Score:", best_grid['n_clusters'])
    # print("Best number of clusters based on the Elbow Method:", best_inertia_grid['n_clusters'])

    return best_grid['n_clusters']


def calculate_avg(data):
    """
    This function takes in a dataframe containing only the values for a specified polymer so that the values can be
    averaged to provide a "reference point" for distance calculations
    """

    """Defining the ranges of rows to be averaged separately for the specific DataFrame"""
    ranges = [(0, 6), (7, 13), (14, 19), (20, 26), (27, 33), (34, 40), (41, 41)]
    avg_names = ["2055", "2056", "2057", "2058", "2059", "2060", "30035"]

    """Initializing an array to store the averaged column values and standard deviations"""
    averaged_dataset = []
    one_std_dataset = []

    for start, end in ranges:
        """Using .iloc for DataFrame slicing"""
        block = data.iloc[start:end + 1, :]
        averaged_dataset.append(block.mean(axis=0).to_frame().T)
        one_std_dataset.append(block.std(axis=0).to_frame().T)

    result = pd.concat([pd.concat(averaged_dataset, ignore_index=True), pd.concat(one_std_dataset, ignore_index=True)],
                       axis=1)
    result.columns = [f'{col}_{stat}' for stat in ['Avg', 'Std'] for col in data.columns]

    return result, avg_names


def initial_data_graphs(concatenated_data, frequency_data, dataset_name):
    """
    This function takes in the data DataFrame which includes all the storage and loss moduli as well as the viscosity
    data and the frequency Dataframe to produce graphs of each of the features in relation to the frequency. Only a
    figure is produced in this function and there is nothing returned to main.
    """

    """Defining data and labels"""
    data_sets = ["G' [Pa]", "G'' [Pa]", "|n*| [Pa]"]
    legend_labels = ["2055_.1", "2055_.2", "2055_.3", "2055_.4", "2055_.5", "2055_.6", "2055_.7", "2056_.1", "2056_.2",
                     "2056_.3", "2056_.4", "2056_.5", "2056_.6", "2056_.7", "2057_.1", "2057_.2", "2057_.3", "2057_.5",
                     "2057_.6", "2057_.7", "2058_.1", "2058_.2", "2058_.3", "2058_.4", "2058_.5", "2058_.6", "2058_.7",
                     "2059_.1", "2059_.2", "2059_.3", "2059_.4", "2059_.5", "2059_.6", "2059_.7", "2060_.1", "2060_.2",
                     "2060_.3", "2060_.4", "2060_.5", "2060_.6", "2060_.7", "30035_.1", "30035_.2", "30035_.3",
                     "30035_.4", "30035_.5", "30035_.6", "30035_.7"]
    freq1 = frequency_data.iloc[0][0:16].tolist()

    """Creating subplots"""
    fig, axs = plt.subplots(3, sharex=True, sharey=True, figsize=(10, 6))
    """Adjusting the vertical spacing"""
    plt.subplots_adjust(hspace=0.5, right=0.8)

    """Commented out one is for single polymer"""
    # line_styles = ['-', ':', '--', '']
    line_styles = ['-', ':', '--', '-.', '-', ':', '--']

    for i, ax in enumerate(axs):
        ax.set_title(f"{data_sets[i]} VS Freq")
        ax.set_ylabel(data_sets[i])

        if i == 2:
            ax.set_xlabel('Frequency [Hz]')

        for j in range(47):
            data = concatenated_data.iloc[j][i * 14:(i + 1) * 14].tolist()
            """Using different line styles for every 7 plots"""
            line_style = line_styles[j // 7]
            """Assigning a different color for every seventh line"""
            color = f'C{j % 7}'
            ax.plot(freq1, data, label=legend_labels[j], linestyle=line_style, color=color)

    """Commented out one is for single polymer"""
    # ax.legend(legend_labels, loc="center left", bbox_to_anchor=(1, 3.2), fontsize='small', ncol=2)
    ax.legend(legend_labels, loc="upper left", bbox_to_anchor=(1, 3.2), fontsize='small', ncol=2)

    plt.show()

    save_plot(fig, f"Standardized Data.png", dataset_name,
              os.path.join(r"C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running Data\Last Run"))


def run_clustering_analysis(data, num_components_range, sample_names, dataset_name):
    """
    Creating a random seed for the initialization of cluster centroids. This allows for more consistent and repeatable
    results. This basically just creates a constant to initialize the random number generator for k-means clustering.
    The value can be replaced by different integer values
    """

    random_seed = 150
    np.random.seed(random_seed)

    """Creating arrays to store the optimum cluster, variance, pca weights and cluster assignments for each PCA test"""
    optimized_clusters = []
    variance = []
    pca_components_list = []
    cluster_data = []

    """Creating a DataFrame to store the sample names for cluster assignments"""
    pca_cluster_data = pd.DataFrame({
        'Sample Names': sample_names,
    })

    """This loops through all of the various PCA tests for the original dataset"""
    with pd.ExcelWriter(
            f'C:\\Users\\tnt02\\OneDrive\\Desktop\\Masters Research\\Running Data\\Last Run\\Excel '
            f'Files\\{dataset_name}_PCA_Cluster_Assignments.xlsx', engine='xlsxwriter') as writer:

        for num_components in num_components_range:

            pca_result, cumulative_variance = pca_embeddings(data, dataset_name, num_components=num_components)

            """Checking for variance - we only want over 90% variance since the other PCA components do not provide
            a good model to base results off of"""
            if cumulative_variance[-1] * 100 < 90:
                continue

            """Printing PCA results for Original Dataset"""
            # print(f"Explained Variance (PCA {num_components}): {cumulative_variance[-1] * 100:.2f}%")
            # print("Top Principal Components (Original Data:")
            # print(pca_result)
            # print("")

            optimum_num_clusters = kmean_hyper_param_tuning(pca_result, num_components, dataset_name)
            # print("Optimum num of clusters based on silhouette scores =", optimum_num_clusters)

            """Fitting KMeans"""
            kmeans = KMeans(n_clusters=optimum_num_clusters, random_state=random_seed)
            kmeans.fit(pca_result)

            """Adding cluster assignments for the current PCA component"""
            pca_cluster_data[f'Cluster (PCA {num_components})'] = kmeans.labels_

            """Appending each result"""
            pca_components_list.append(num_components)
            # print(pca_components_list)

            optimized_clusters.append(optimum_num_clusters)
            variance.append(cumulative_variance[-1] * 100)

            """Storing cluster assignments for the current PCA component"""
            cluster_data.append(kmeans.labels_)

            individual_cluster_data = pd.DataFrame({
                'Sample Names': sample_names,
            })

            # print(f'\nCluster assignments for PCA {num_components}:')

            """Sorting the cluster data by cluster assignment"""
            individual_cluster_data['Cluster Assignments'] = kmeans.labels_
            individual_cluster_data_sorted = individual_cluster_data.sort_values(by='Cluster Assignments')
            # print(individual_cluster_data_sorted)
            # print("")

            sheet_name = f'PCA_{num_components}_CLuster_Assignments'
            individual_cluster_data_sorted.to_excel(writer, sheet_name=sheet_name, index=False)

    """Creating Table to determine the best number of PCA components to use"""
    table_data = {
        'PCA Components': pca_components_list,
        'Optimized Clusters': optimized_clusters,
        'Variance Explained (%)': variance
    }

    """Putting the PCA result into a dataframe to export the results to an Excel file"""
    top_PCs = pd.DataFrame(pca_result)

    top_PCs.to_excel(r'C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running Data\Last Run\Excel '
                     r'Files\Top_PCs.xlsx', sheet_name='sheet1', index=False)

    """Sorting the cluster data by cluster assignment"""
    pca_cluster_data['Cluster Assignments'] = kmeans.labels_

    """ Saving the clustering results in tables and producing a heatmap for visual aid"""
    save_tables(table_data, pca_cluster_data, dataset_name,
                os.path.join(r"C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running Data\Last Run\Excel Files"))
    heatmap(cluster_data, sample_names, pca_components_list)

    return top_PCs


def calculate_distances(data, datapoint, distance_metric='euclidean'):
    """
    This function takes in the dataset the names (polymers) in the dataset, the datapoint of interest and what distance
    metric to use to calculate the distance between the specified datapoint and the other datapoints in the clustering.
    This function returns a DataFrame which includes all the distances between the polymers and the names associated
    with the distances. The output of this function is placed in a table to determine what polymers are the closest to
    the specified datapoint.
    """

    """Creating an array to store all the distance values"""
    distances = []

    for i, (index, row) in enumerate(data.iterrows()):
        distance = 0

        if distance_metric == 'euclidean':
            distance = euclidean(row, datapoint)
        elif distance_metric == 'cityblock':
            distance = cityblock(row, datapoint)
        elif distance_metric == 'cosine':
            distance = cosine(row, datapoint)

        distances.append(distance)

    return pd.DataFrame(distances)


def heatmap(cluster_data, sample_names, pca_components_list):
    """
    This function takes in the data from the clustering, the names of the polymers, and the number of PCs used and
    creates a heatmap for between visualization of the clustering results using the imported module Plotly Express
    """

    cluster_data = pd.DataFrame(cluster_data).T  # Transpose cluster data

    fig = px.imshow(cluster_data, color_continuous_scale='viridis')
    fig.update_layout(
        title="Cluster Assignments for Different PCA Components",
        xaxis_title="PCA Components",
        yaxis_title="Sample",
        yaxis_tickvals=list(range(len(sample_names))),
        yaxis_ticktext=sample_names,
        xaxis_tickvals=list(range(len(pca_components_list))),
        xaxis_ticktext=[f"PCA {num}" for num in pca_components_list],
    )

    fig.show()


def pearson_correlation(pca_result, eds_data):
    """
    This function calculates the correlation between the various PCA results and Ed's calculations (viscosity,
    crossover, frequency). In addition to calculating the correlation each of the corresponding datapoints are plotted.
    """

    # NEED TO FIGURE OUT HOW TO DEAL WITH NaN VALUES!!!


    correlations = pd.DataFrame(index=[f'PCA{i + 1}' for i in range(len(pca_result.columns))], columns=eds_data.columns)

    for col_pca in pca_result.columns:

        for col_eds in eds_data.columns:

            """Calculating the correlation that corresponds to the graph"""
            correlation = np.corrcoef(pca_result[col_pca], eds_data[col_eds])[0, 1]
            correlations.loc[f'PCA{pca_result.columns.get_loc(col_pca) + 1}', col_eds] = correlation

            """Creating a scatter plot with the PCA column values as x values and the eds data column values as the y
            """
            plt.figure()
            plt.scatter(pca_result[col_pca], eds_data[col_eds])
            plt.title(f'Scatter Plot: PCA{pca_result.columns.get_loc(col_pca) + 1} vs {col_eds}')
            plt.xlabel(f'PCA{pca_result.columns.get_loc(col_pca) + 1}')
            plt.ylabel(col_eds)
            plt.text(0.6, 0.9, f'Correlation: {correlation: .2f}', transform=plt.gca().transAxes, fontsize=10,
                     color='blue')

            """Making sure that there are no invalid characters in the filename"""
            scatter_filename = f'scatter_PCA{pca_result.columns.get_loc(col_pca) + 1} _vs_{col_eds}'
            sanitized_filename = sanitize_filename(scatter_filename)

            save_plot(plt, sanitized_filename, dataset_name="Original", folder_path=r"C:\Users\tnt02"
                                                                                    r"\OneDrive"
                                                                                    r"\Desktop\Masters "
                                                                                    r"Research\Running "
                                                                                    r"Data\Last "
                                                                                    r"Run\Correlations")
            plt.close()

    return correlations


def main():

    """Suppressing Warning Signs"""
    warnings.simplefilter(action='ignore', category=FutureWarning)  # This warning is because a variable changes
    warnings.simplefilter(action='ignore', category=RuntimeWarning)  # This warning is for generating 20+ pictures
    warnings.simplefilter(action='ignore', category=UserWarning)  # This warning occurs do to tight layout for some PCAs

    print("Loading Dataset")
    sample_names, concatenated_data, frequency_data, eds_data = load_embeddings()

    """Naming the 2 different datasets"""
    data1 = "Original"

    """Graphing storage and loss moduli and viscosity as a function of frequency"""
    initial_data_graphs(concatenated_data, frequency_data, data1)

    """Determining the maximum number of components based on the minimum of samples and dimensions and the range of PCA 
           components to iterate over for the original dataset (from 1 to the max number of components"""
    max_components_original = min(concatenated_data.shape[0], concatenated_data.shape[1])
    num_components_range_original = range(1, max_components_original + 1)

    """Need the clustering function to return the pca_result so that distance calculations can be made"""
    print("Running Clustering Analysis on Original Dataset:")
    pca_result = run_clustering_analysis(concatenated_data, num_components_range_original, sample_names, data1)

    """3 PCs is sufficient for this dataset so need to take the first 3 columns of the pca_result for the distance
    calculations"""
    pca_result = pca_result.iloc[:, :3]
    # print('\nPCA Values Pulled:')
    # print(pca_result)

    """Need to average the points pertaining to lot 30035 to obtain a reference datapoint"""
    reference_point_data = pca_result.iloc[41:48, :]
    reference_point = reference_point_data.mean(axis=0).to_frame().T
    resulting_averaged_data = pd.concat([pca_result[0:41], reference_point], ignore_index=True)
    # print('\nPCA Results with Averaged 30035 Reference Point:')
    # print(resulting_averaged_data)

    """Need to make distance calculations from the lot 30035 data point to each of the different polymer tests"""
    datapoint = resulting_averaged_data.iloc[41]  # Pulling lot 30035 and setting as reference point
    distances = calculate_distances(resulting_averaged_data, datapoint, distance_metric='euclidean')

    """Averaging the distances for the same test"""
    averaged_distances, avg_names = calculate_avg(distances)

    distance_table = pd.DataFrame({'Lot': avg_names, 'Distance to 30035': averaged_distances['0_Avg'].values,
                                   'Uncertainty': averaged_distances['0_Std'].values})
    distance_table.columns = ['Lot', 'Distance to 30035', 'Uncertainty']
    distance_table = distance_table.sort_values(by='Distance to 30035')
    print("\nDistance from 30035:")
    print(distance_table)

    """Saving the DataFrame to an Excel file"""
    distance_table.to_excel(r'C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running Data\Last Run\Excel '
                            r'Files\Euclidean_Distance_Table.xlsx', sheet_name='sheet1', index=False)

    """Calculating pearson correlation values between PCA results and Ed's calculations"""
    correlations = pearson_correlation(pca_result, eds_data)
    print("\n Correlations: ")
    print(correlations)

    """Saving the DataFrame to an Excel file"""
    correlations.to_excel(r'C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running Data\Last Run\Excel '
                          r'Files\Pearson_Correlation_Coefficients.xlsx', sheet_name='sheet1', index=False)


if __name__ == "__main__":
    main()
