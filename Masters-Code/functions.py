"""Importing packages"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import os
import re

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean, cityblock, cosine


def load_embeddings_rheology():
    """
    This function takes in an Excel file with multiple sheets of rheology data. The first sheet contains pre-calculated
    data for later comparisons. These calculations are saved in a dataframe called eds_data. The rest of the sheets
    contain the raw rheology data for different tests. Only columns D through G were pulled. Column D included the
    frequency data and was stored in a separate dataframe while the rest of the columns where transposed and
    concatenated. These columns contain data for viscosity and storage and loss moduli. The sample names are also saved
    in their own dataframe and outputted. This function also standardizes the data.
    """

    """Loading excel file"""
    raw_data = "High Precision 2055-2060 Series 170C.xlsx"
    # raw_data = "2055-2060 Series 70C.xlsx"
    # raw_data = "2055-2060 Series 120C.xlsx"
    # raw_data = "2055-2060 Series 170C.xlsx"

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

        if index == 0:  # Skip the first sheet
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
        df = df.iloc[0:29, :]

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
        # concatenated_row = g_prime + g_double_prime

        frequencies_rows.append(frequencies_row)
        concatenated_rows.append(concatenated_row)

        """Storing the sheet name as the sample name"""
        label = extract_label(sheet_name)
        sample_names.append(label)

    """Creating a DataFrame with the concatenated rows"""
    concatenated_data = pd.DataFrame(concatenated_rows)
    print("\nRaw Data:")
    print(concatenated_data)
    print("")

    """HERE ARE THE STANDARDSCALERS!"""
    scaler = StandardScaler(with_std=True)
    # scaler = StandardScaler()

    """Applying standardscaler to the dataset"""
    concatenated_data_scaled = scaler.fit_transform(concatenated_data)
    concatenated_data_df = pd.DataFrame(concatenated_data_scaled, columns=concatenated_data.columns)
    print("Data After Standardizing:")
    print(concatenated_data_df)
    print("")

    frequency_data = pd.DataFrame(frequencies_rows)
    print(frequency_data)

    num_datapoints = len(g_prime)

    return sample_names, concatenated_data_df, frequency_data, eds_data, num_datapoints


def load_embeddings_tensile():
    """
    This function takes in an Excel file with multiple sheets for tensile testing data. When data is read in a sheet is
    specified for a certain temperature. This function can standardize the data. It returns 2 dataframes one with the
    data itself and the other with the names of the polymers.
    """

    """Loading excel file"""
    # raw_data = "Tensile Data 2055-2060 24C.xlsx"
    # raw_data = "Tensile Data 2055-2060 40C.xlsx"
    # raw_data = "Tensile Data 2055-2060 50C.xlsx"
    raw_data = "Tensile Data 2055-2060 70C.xlsx"

    """Creating ranges for the different files"""
    # original_ranges = [(0, 3), (4, 10), (11, 17), (18, 24), (25, 30), (31, 36), (37, 43)]  # 24C
    # original_ranges = [(0, 5), (6, 11), (12, 17), (18, 23), (24, 29), (30, 36), (37, 43)]  # 40C
    # original_ranges = [(0, 5), (6, 12), (13, 19), (20, 25), (26, 32), (33, 38), (39, 45)]  # 50C
    original_ranges = [(0, 5), (6, 12), (13, 19), (20, 24), (25, 29), (30, 35)]  # 70C

    """Creating an ExcelFile object to read the Excel file"""
    xls = pd.ExcelFile(raw_data)

    """Getting a list of all the sheet names (multiple sheets in the excel file)"""
    sheet_names = xls.sheet_names

    """Creating empty lists to store individual data DataFrames and sample names"""
    sample_names = []
    all_stress = []
    all_strain = []
    all_force = []
    # all_displacement = []

    """Looping through each sheet and reading data into a dataframe"""
    for index, sheet_name in enumerate(sheet_names):

        if index == 0:  # Skip the first sheet
            print("Ed's Calculations: ")
            eds_data = pd.read_excel(xls, sheet_name=sheet_name, usecols="C:E")
            print(eds_data)
            continue

        """Storing the sheet name as the sample name"""
        label = extract_label(sheet_name)
        sample_names.append(label)

        """Reading in the data from the current sheet into a DataFrame we want only values pertaining to columns D
                through G since indexing starts at 0 column D is index 3 and so on"""
        df = pd.read_excel(xls, sheet_name=sheet_name, usecols="A:F")

        """Need to take the actual data which starts at row 13 of Excel sheet"""
        df = df.iloc[13:, :]

        """Placing all the values for displacement, stress, and strain in their own lists so that they can be added 
        into the DataFrames"""
        stress = df.iloc[:, 0].tolist()
        strain = df.iloc[:, 1].tolist()
        force = df.iloc[:, 2].tolist()
        # displacement = df.iloc[:, 4].tolist()

        all_stress.append(stress)
        all_strain.append(strain)
        all_force.append(force)
        # all_displacement.append(displacement)

    """Creating a DataFrame with the concatenated rows"""
    stress_data = pd.DataFrame(all_stress)
    strain_data = pd.DataFrame(all_strain)
    force_data = pd.DataFrame(all_force)
    # displacement_data = pd.DataFrame(all_displacement)

    num_datapoints = len(stress)

    """HERE ARE THE STANDARDSCALERS!"""
    # scaler = StandardScaler(with_std=True)
    # scaler = StandardScaler()

    """Applying standardscaler to the dataset"""
    # stress_data_scaled = scaler.fit_transform(stress_data)
    # stress_data = pd.DataFrame(stress_data_scaled, columns=stress_data.columns)
    # print("Data After Standardizing:")
    # print(stress_data)
    # print("")

    return sample_names, stress_data, strain_data, force_data, num_datapoints, eds_data, original_ranges


def load_embeddings_combined():
    """
    This function is just like load_embeddings_rheology. The only difference in this function is that the tensile data
    are added to the dataset at the end of each row for each polymer
    """

    """Loading excel file"""
    raw_data = "Combined Data Series 2055-2060 (70T).xlsx"

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
        df = pd.read_excel(xls, sheet_name=sheet_name, usecols=[3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 19])

        """Need to take only values from rows 2 through 14 since there are NaN values for the first 7 sheets
        PCA does not support NaN values, so I truncated the data for the last files that had additional data
        (This was implemented for a different dataset, however, it works for this dataset)"""
        # df = df.iloc[0:14, :]

        """Placing all the values for frequency, storage modulus (G'), loss modulus (G''), and viscosity in their own
        lists so that they can be added into the DataFrames"""
        frequencies_row = df.iloc[:, 0].tolist()
        g_prime = df.iloc[:, 1].tolist()
        g_double_prime = df.iloc[:, 2].tolist()
        viscosity = df.iloc[:, 3].tolist()
        tensile_data = df.iloc[0, 4:10].values.tolist()
        # tensile_data = df.iloc[0, [4, 6, 8]].values.tolist()
        # tensile_data_dev = df.iloc[0, [5, 7, 9]].values.tolist()

        """HERE ARE THE LOGGED VALUES!"""
        # g_prime = [np.log10(value) for value in g_prime]
        # g_double_prime = [np.log10(value) for value in g_double_prime]
        # viscosity = [np.log10(value) for value in viscosity]
        # tensile_data = [np.log10(value) for value in tensile_data[:-1]]
        # tensile_data_dev = [np.log10(value) for value in tensile_data_dev[:-1]]

        """Concatenating the standardized values. HERE is where I add in the tensile testing data to the end of the 
        row"""
        concatenated_row = g_prime + g_double_prime + viscosity + tensile_data
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


def extract_label(sheet_name):
    # Use regular expression to find the numbers in the sheet name
    match = re.search(r'\d+', sheet_name)
    if match:
        label = match.group()
        return label
    else:
        return None


def pca_embeddings(data, dataset_name, num_components):
    """
    This function takes in the scaled DataFrame (data) and the number of components to iterate over for the data. Inside
    the function the PCA histograms that show how PCA weighted each feature in the dataset. This function also produces
    the cumulative variance plots, so we can see how increasing number of PCs contributes to increased captured variance
    in the dataset. The function then returns the pca results (pca weighted coefficients) and the cumulative variance
    """

    """This is where PCA is actually applied to the dataset"""
    pca = PCA(n_components=num_components)  # Defining how many PCs to use
    result = pca.fit_transform(data)  # Transforming the data to the specified PCs to use

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
        save_plot(fig, f"Weights_{num_components}.png", dataset_name,
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
        save_plot(fig_percent_variance, f"Percent_Variance_{num_components}.png", dataset_name,
                  os.path.join(
                      r"C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running Data\Last Run\Percent Variance"))

    return result, cumulative_variance


def lda_embeddings(data, sample_names, dataset_name, num_components):
    """
    This function takes in the scaled DataFrame (data) and the number of components to iterate over for the data. Inside
    the function the LDA histograms that show how LDA weighted each feature in the dataset. This function also produces
    the cumulative variance plots, so we can see how increasing number of components contributes to increased captured
    variance in the dataset. The function then returns the lda results and the cumulative variance. The only difference
    between the use of LDA and PCA is that LDA classifies the data based on the known polymer type and then calculates
    the variance between the different polymers so that the variance between same polymers is not modeled.
    """

    # print("Here is the lda_embeddings data: \n", data)
    # print("Here are the lda_embeddings sample names: \n", sample_names)

    """This is where LDA is actually applied to the dataset"""
    lda = LinearDiscriminantAnalysis(n_components=num_components)  # Defining how many components
    result = lda.fit_transform(data, sample_names)  # Transforming the data
    # print(result.T)
    num_features = data.shape[1]

    # print('\nThis is LDA Result:\n', result)
    # print('\nThis is LDA Coef: \n', lda.coef_[0])

    lda_mean_embeddings = lda.means_
    lda_mean_embeddings_df = pd.DataFrame(lda_mean_embeddings)
    # print("Here is the mean for lda embeddings:", lda_mean_embeddings)

    lda_mean_embeddings_df.to_excel(r'C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running Data\Last Run\Excel '
                                    r'Files\Mean Vectors LDA embeddings.xlsx', sheet_name='sheet1', index=False)

    eigenvectors = lda.scalings_
    # print(eigenvectors.shape)
    eigenvalues = lda.explained_variance_ratio_

    """Need to transpose eigenvectors should be a 6x42 not 42x6"""
    eigenvectors = eigenvectors.T

    # print("Eigenvalues:", eigenvalues)
    # print("Eigenvectors:", eigenvectors)
    # print(eigenvectors.shape)

    eigenvectors_df = pd.DataFrame(eigenvectors)

    eigenvectors_df.to_excel(r'C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running Data\Last Run\Excel '
                             r'Files\Eigenvectors LDA embeddings.xlsx', sheet_name='sheet1', index=False)

    # data_2 = np.array(data)
    # first_point = np.matmul(data_2, eigenvectors.T)
    # print(first_point[:, 0:2])
    # print(first_point.shape)

    """Checking if there's only one component (PC) because then we do not use subplots. If there is more than one 
        component then subplots have to be made so that we can see how each feature in the component (PC) was weighted"""
    if num_components == 1:
        """Creating the histogram"""
        fig, ax = plt.subplots()
        ax.bar(range(num_features), eigenvectors[0], tick_label=data.columns)
        ax.set_title("LDA 1 Component Analysis")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Weight")
        plt.xticks(rotation=90)

    else:
        """Creating subplots for PCA component analysis"""
        fig, axes = plt.subplots(num_components, 1, sharex=True)

        """Creating the histograms"""
        for i, ax in enumerate(axes):
            ax.bar(range(num_features), eigenvectors[i], tick_label=data.columns)
            ax.set_title(f"LDA {i + 1} Component Analysis")
            ax.set_xlabel("Feature")
            ax.set_ylabel("Weight")
            plt.xticks(rotation=90)

        plt.tight_layout()

    """Only saving and the first 5 LDAs since it becomes unhelpful with more"""
    if num_components <= 5:
        """Calling to save function"""
        save_plot(fig, f"Weights_{num_components}.png", dataset_name,
                  os.path.join(r"C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running Data\Last Run\PCA Analysis"))

    """Calculating variance ratios"""
    variance = lda.explained_variance_ratio_
    # print("\nThese are the explained variance ratios: ", variance)
    cumulative_variance = variance.cumsum()

    """Creating the variance percentage plot"""
    fig_percent_variance, ax_percent_variance = plt.subplots(figsize=(8, 4))
    ax_percent_variance.bar(range(1, num_components + 1), variance * 100, alpha=0.5, align='center', label='Variance')
    ax_percent_variance.step(range(1, num_components + 1), cumulative_variance * 100, where='mid',
                             label='Cumulative Variance')
    ax_percent_variance.set_title('Variance Ratios For LDA Components')
    ax_percent_variance.set_xlabel('LDA')
    ax_percent_variance.set_ylabel('Variance (%)')
    ax_percent_variance.legend(loc='best')

    """Only saving and the first 5 LDAs since it becomes unhelpful with more"""
    if num_components <= 5:
        """Calling to save function"""
        save_plot(fig_percent_variance, f"Percent_Variance_{num_components}.png", dataset_name,
                  os.path.join(r"C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running Data\Last Run\Percent "
                               r"Variance"))

    return result, cumulative_variance


def isomap_embeddings(data, num_components):

    # Applying Isomap
    # Setting n_neighbors to 7 since there are 7 types of tests (this may be wrong)
    isomap = Isomap(n_components=num_components)
    isomap_results = isomap.fit_transform(data)
    variances = np.var(isomap_results, axis=0)
    cumulative_variance_isomap = variances / np.sum(variances)

    # Applying T-SNE

    return isomap_results, cumulative_variance_isomap


def lle_embeddings(data, num_components):

    # Applying LLE
    lle = LocallyLinearEmbedding(n_components=num_components)
    lle_results = lle.fit_transform(data)
    variances = np.var(lle_results, axis=0)
    cumulative_variance_lle = variances / np.sum(variances)

    return lle_results, cumulative_variance_lle


def tsne_embeddings(data, num_components):

    # Applying TSNE
    tsne = TSNE(n_components=num_components)
    tsne_results = tsne.fit_transform(data)
    variances = np.var(tsne_results, axis=0)
    cumulative_variance_tsne = variances / np.sum(variances)

    return tsne_results, cumulative_variance_tsne


def initial_data_graphs_rheology(data, frequency_data, dataset_name, num_datapoints):
    """
    This function is only used for rheology data and it takes in the data DataFrame which includes all the storage and
    loss moduli as well as the viscosity data and the frequency Dataframe to produce graphs of each of the features in
    relation to the frequency. Only a figure is produced in this function and there is nothing returned to main.
    """

    """Defining data and labels"""
    data_sets = ["G' [Pa]", "G'' [Pa]", "|n*| [Pa]"]
    legend_labels = ["2055_.1", "2055_.2", "2055_.3", "2055_.4", "2055_.5", "2055_.6", "2055_.7", "2056_.1", "2056_.2",
                     "2056_.3", "2056_.4", "2056_.5", "2056_.6", "2056_.7", "2057_.1", "2057_.2", "2057_.3", "2057_.5",
                     "2057_.6", "2057_.7", "2058_.1", "2058_.2", "2058_.3", "2058_.4", "2058_.5", "2058_.6", "2058_.7",
                     "2059_.1", "2059_.2", "2059_.3", "2059_.4", "2059_.5", "2059_.6", "2059_.7", "2060_.1", "2060_.2",
                     "2060_.3", "2060_.4", "2060_.5", "2060_.6", "2060_.7", "30035_.1", "30035_.2", "30035_.3",
                     "30035_.4", "30035_.5", "30035_.6", "30035_.7"]

    freq1 = frequency_data.iloc[0][0:num_datapoints].tolist()

    """Creating subplots"""
    fig, axs = plt.subplots(3, sharex=True, figsize=(10, 6))
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
            data_values = data.iloc[j][i * num_datapoints:(i + 1) * num_datapoints].tolist()
            """Using different line styles for every 7 plots"""
            line_style = line_styles[j // 7]
            """Assigning a different color for every seventh line"""
            color = f'C{j % 7}'
            ax.plot(freq1, data_values, label=legend_labels[j], linestyle=line_style, color=color)

    """Commented out one is for single polymer"""
    # ax.legend(legend_labels, loc="center left", bbox_to_anchor=(1, 3.2), fontsize='small', ncol=2)
    ax.legend(legend_labels, loc="upper left", bbox_to_anchor=(1, 3.2), fontsize='small', ncol=2)

    plt.show()

    save_plot(fig, f"Standardized Data.png", dataset_name,
              os.path.join(r"C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running Data\Last Run"))


def initial_data_graphs_high_res_rheology(data, frequency_data, dataset_name, num_datapoints, ranges):
    """
    This function is only used for rheology data and it takes in the data DataFrame which includes all the storage and
    loss moduli as well as the viscosity data and the frequency Dataframe to produce graphs of each of the features in
    relation to the frequency. Only a figure is produced in this function and there is nothing returned to main.
    """

    """Defining data and labels"""
    data_sets = ["G' [Pa]", "G'' [Pa]", "|n*| [Pa]"]

    legend_labels = ["30035_.1", "30035_.2", "30035_.3", "30035_.4", "30035_.5", "30035_.6", "30035_.7",
                     "2055_.4", "2055_.5", "2055_.6", "2055_.7",
                     "2056_.1", "2056_.2", "2056_.3", "2056_.4", "2056_.5", "2056_.6", "2056_.7",
                     "2057_.1", "2057_.2", "2057_.3", "2057_.4", "2057_.5", "2057_.6", "2057_.7",
                     "2058_.1", "2058_.2", "2058_.3", "2058_.4", "2058_.5", "2058_.6", "2058_.7",
                     "2059_.1", "2059_.2", "2059_.3", "2059_.4", "2059_.5", "2059_.6", "2059_.7",
                     "2060_.1", "2060_.2", "2060_.3", "2060_.4", "2060_.5", "2060_.6", "2060_.7", ]

    freq1 = frequency_data.iloc[0][0:num_datapoints].tolist()

    """Creating subplots for combined graphs"""
    fig, axs = plt.subplots(3, sharex=True, figsize=(10, 6))
    # fig, axs = plt.subplots(3, sharex=True, figsize=(10, 6))

    """Adjusting the vertical spacing"""
    plt.subplots_adjust(hspace=0.5, right=0.8)

    """Commented out one is for single polymer"""
    marker_styles = [',', 'x', '|', '1', '2', '3', '4']
    colors = ["black", "orange", "blue", "pink", "purple", "red", "yellow"]
    polymer_colors = []
    test_styles = [',', 'x', '|', '1', '2', '3', '4', '1', '2', '3',
                   '4']  # Assigned the line styles for 2055 since doesn't start at 1

    for i, ax in enumerate(axs):
        ax.set_title(f"{data_sets[i]} VS Freq")
        ax.set_ylabel(data_sets[i])

        if i == 2:
            ax.set_xlabel('Frequency [Hz]')

        """Looping through the ranges for polymer lots so that the same polymer lot tests are the same color"""
        for index, (start, end) in enumerate(ranges):
            """Creating a list that contains all the line colors and a list that contains the line styles"""
            polymer_colors.extend([colors[index]] * (end - start + 1))
            polymer_lot = range(start, end + 1)
            for index2, _ in enumerate(polymer_lot):  # Looping within the polymer lot to assign line styles
                if colors[index] == "black" and "orange":  # Skipping 2055 since they were assigned manually
                    continue
                test_styles.append(marker_styles[index2 % len(marker_styles)])

        for j in range(data.shape[0] - 1):
            data_values = data.iloc[j][i * num_datapoints:(i + 1) * num_datapoints].tolist()
            ax.plot(freq1, data_values, label=legend_labels[j], color=polymer_colors[j], linewidth=1,
                    marker=test_styles[j])

    """Commented out one is for single polymer"""
    # ax.legend(legend_labels, loc="lower left", bbox_to_anchor=(1, 3.2), fontsize='small', ncol=2)
    ax.legend(legend_labels, loc="upper left", bbox_to_anchor=(1, 3.2), fontsize='small', ncol=2)

    plt.show()

    save_plot(fig, f"Standardized Data.png", dataset_name,
              os.path.join(r"C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running Data\Last Run"))


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
    if num_components <= 5:
        fig_silhouette, ax_silhouette = plt.subplots()
        ax_silhouette.bar(range(len(silhouette_scores)), list(silhouette_scores), align='center', color='#722f59',
                          width=0.5)
        ax_silhouette.set_xticks(range(len(silhouette_scores)))
        ax_silhouette.set_xticklabels(list(parameters))
        ax_silhouette.set_title('Silhouette Score', fontweight='bold')
        ax_silhouette.set_xlabel('Number of Clusters')

        """Calling to save function"""
        save_plot(fig_silhouette, f"Silhouette_{num_components}.png", dataset_name, os.path.join(
            r"C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running Data\Last Run\Silhouette Scores"))

        """Plotting the elbow graph"""
        fig_elbow, ax_elbow = plt.subplots()
        ax_elbow.plot(parameters, inertias, marker='o', linestyle='-')
        ax_elbow.set_title('Elbow Method for Optimal k')
        ax_elbow.set_xlabel('Number of Clusters (k)')
        ax_elbow.set_ylabel('Inertia')
        ax_elbow.set_xticks(parameters)

        """Calling to save function"""
        save_plot(fig_elbow, f"Elbow_{num_components}.png", dataset_name, os.path.join(
            r"C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running Data\Last Run\Elbow Method"))

        plt.close()

    # print("")
    # print("Best number of clusters based on Silhouette Score:", best_grid['n_clusters'])
    # print("Best number of clusters based on the Elbow Method:", best_inertia_grid['n_clusters'])

    return best_grid['n_clusters']


def run_clustering_analysis(data, num_components_range, sample_names, dataset_name):
    """
    This function performs the clustering analysis for the dataset by applying pca and kmeans. It takes in the dataset,
    the range of PCA's to run over, the names of the polymers, and a dataset name. It returns the pca results and the
    variance captured with each pca. Multiple files are saved in this function.
    """

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
    name_cluster_data = pd.DataFrame({
        'Sample Names': sample_names,
    })

    """This loops through all of the various PCA tests for the original dataset"""
    with pd.ExcelWriter(
            f'C:\\Users\\tnt02\\OneDrive\\Desktop\\Masters Research\\Running Data\\Last Run\\Excel '
            f'Files\\{dataset_name}_Cluster_Assignments.xlsx', engine='xlsxwriter') as writer:

        for num_components in num_components_range:

            if dataset_name == "LDA":
                result, cumulative_variance = lda_embeddings(data, sample_names, dataset_name,
                                                             num_components=num_components)
            elif dataset_name == "PCA":
                result, cumulative_variance = pca_embeddings(data, dataset_name, num_components=num_components)

            elif dataset_name == "ISO":
                result, cumulative_variance = isomap_embeddings(data, num_components=num_components)

            elif dataset_name == "LLE":
                result, cumulative_variance = lle_embeddings(data, num_components=num_components)

            elif dataset_name == "TSNE":
                result, cumulative_variance = tsne_embeddings(data, num_components=num_components)

            """Checking for variance - we only want over 90% variance since the other PCA components do not provide
            a good model to base results off of"""
            # if cumulative_variance[-1] * 100 < 90:
            #   continue

            optimum_num_clusters = kmean_hyper_param_tuning(result, num_components, dataset_name)
            # print("Optimum num of clusters based on silhouette scores =", optimum_num_clusters)

            """Fitting KMeans"""
            kmeans = KMeans(n_clusters=3, random_state=random_seed)  # CHANGE NUM OF CLUSTERS USED HERE
            kmeans.fit(result)

            """Adding cluster assignments for the current PCA component"""
            name_cluster_data[f'Cluster (Dim_{num_components})'] = kmeans.labels_

            """Appending each result"""
            pca_components_list.append(num_components)

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

            sheet_name = f'Dim_{num_components}_Cluster_Assignments'
            individual_cluster_data_sorted.to_excel(writer, sheet_name=sheet_name, index=False)

    """Creating Table to determine the best number of PCA components to use"""
    table_data = {
        'PCA Components': pca_components_list,
        'Optimized Clusters': optimized_clusters,
        'Variance Explained (%)': variance
    }

    """Putting the PCA result into a dataframe to export the results to an Excel file"""
    results = pd.DataFrame(result)

    results.to_excel(r'C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running Data\Last Run\Excel '
                     r'Files\Results.xlsx', sheet_name='sheet1', index=False)

    """Sorting the cluster data by cluster assignment"""
    name_cluster_data['Cluster Assignments'] = kmeans.labels_

    """ Saving the clustering results in tables and producing a heatmap for visual aid"""
    save_tables(table_data, name_cluster_data, dataset_name,
                os.path.join(r"C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running Data\Last Run\Excel Files"))

    heatmap(cluster_data, sample_names, pca_components_list)

    return results, cumulative_variance


def plot_2D(sample_names, result, dataset_name, ranges):
    """Plotting the first two components"""

    print(dataset_name)
    colors = ["black", "orange", "blue", "pink", "purple", "red", "yellow"]
    names = []

    com_1 = result.loc[:, 0]
    com_2 = result.loc[:, 1]

    """Checking to see which dataset we are working with"""
    if dataset_name == "PCA":
        plt.figure()
        for index, (start, end) in enumerate(ranges):
            index_class = range(start, end + 1)
            plt.scatter(com_1.iloc[index_class], com_2.iloc[index_class], color=colors[index],
                        label=f'Class: {sample_names[index]}')

        plt.legend(loc="upper left", bbox_to_anchor=(1, 0.5), shadow=False, scatterpoints=1)
        plt.title(f"2D {dataset_name} Results")
        plt.xlabel(f"{dataset_name} Component 1")
        plt.ylabel(f"{dataset_name} Component 2")

        # plt.xlim(-20, 30)
        # plt.ylim(-5, 7)

        # plt.xlim(-100, 100)
        # plt.ylim(-60, 60)

    else:
        plt.figure()
        for index, (start, end) in enumerate(ranges):
            index_class = range(start, end + 1)
            plt.scatter(com_1.iloc[index_class], com_2.iloc[index_class], color=colors[index],
                        label=f'Class: {sample_names[index]}')

        plt.legend(loc="upper left", bbox_to_anchor=(1, 0.5), shadow=False, scatterpoints=1)
        plt.title(f"2D {dataset_name} Results")
        plt.xlabel(f"{dataset_name} Component 1")
        plt.ylabel(f"{dataset_name} Component 2")

        # plt.xlim(-70, 110)
        # plt.ylim(-30, 50)

        # plt.xlim(-20, 40)
        # plt.ylim(-15, 25)

    save_plot(plt, f"2D {dataset_name} Results.png", dataset_name,
              os.path.join(r"C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running Data\Last Run"))

    plt.close()


def heatmap(cluster_data, sample_names, components_list):
    """
    This function takes in the data from the clustering, the names of the polymers, and the number of PCs used and
    creates a heatmap for between visualization of the clustering results using the imported module Plotly Express
    """

    cluster_data = pd.DataFrame(cluster_data).T  # Transpose cluster data

    fig = px.imshow(cluster_data, color_continuous_scale='viridis')
    fig.update_layout(
        title="Cluster Assignments for Different Dimensionality",
        xaxis_title="Dimensions",
        yaxis_title="Sample",
        yaxis_tickvals=list(range(len(sample_names))),
        yaxis_ticktext=sample_names,
        xaxis_tickvals=list(range(len(components_list))),
        xaxis_ticktext=[f"{num}" for num in components_list],
    )

    fig.show()


def calculate_avg(data, ranges, avg_names):
    """
    This function takes in a dataframe containing only the values for a specified polymer so that the values can be
    averaged to provide a "reference point" for distance calculations
    """

    """Initializing an array to store the averaged column values and standard deviations"""
    averaged_dataset = []
    one_std_dataset = []

    for start, end in ranges:
        if data.shape[1] == 1:  # Check if there is only one column in the DataFrame
            block = data.iloc[start:end + 1].values  # Using .values to get the values to average
        else:
            block = data.iloc[start:end + 1, :]

        avg_block = np.mean(block, axis=0)
        std_block = np.std(block, axis=0)

        avg_block_df = pd.DataFrame(data=avg_block, columns=data.columns if data.shape[1] > 1 else [data.columns[0]])
        std_block_df = pd.DataFrame(data=std_block, columns=data.columns if data.shape[1] > 1 else [data.columns[0]])

        averaged_dataset.append(avg_block_df)
        one_std_dataset.append(std_block_df)

    result = pd.concat(
        [pd.concat(averaged_dataset, ignore_index=True),
         pd.concat(one_std_dataset, ignore_index=True)],
        axis=1
    )

    result.columns = [f'{col}_{stat}' for stat in ['Avg', 'Std'] for col in data.columns]
    # print(result)

    return result, avg_names


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


def pearson_correlation(result, eds_data, avg_names, cumulative_variance, dev_data, dataset_name, original_ranges):
    """
    This function calculates the correlation between the various PCA results and Ed's calculations (viscosity,
    crossover, frequency).
    """

    correlations = pd.DataFrame(index=[f'PCA_{i + 1}' for i in range(len(result.columns))], columns=eds_data.columns)
    all_average_results = pd.DataFrame()
    all_error_results = pd.DataFrame()

    """Creating new a new set of ranges that can be updated as to keep the original ranges intact"""
    modified_ranges = original_ranges.copy()

    for col_result in result.columns:

        current_column_result = result[col_result]

        """Calculating the variance explained for the current PCA and calling the averaged_histogram"""
        if col_result == 0:  # setting the current variance to the first value in the cumulative_variance array
            current_variance = cumulative_variance[col_result]
        else:  # takes the cumulative_variance at the indexed value and subtracts the previous indices
            current_variance = cumulative_variance[col_result] - cumulative_variance[col_result - 1]

        # print("\nThis is the current column at BEFORE of averaged_histogram function:\n", current_column_result)

        """Check for if there was no correlation"""
        averaged_values, error = averaged_histogram(result, current_column_result, col_result, original_ranges,
                                                    avg_names,
                                                    current_variance,
                                                    dev_data, dataset_name)

        all_average_results = pd.concat([all_average_results, averaged_values], ignore_index=True)
        all_error_results = pd.concat([all_error_results, error], ignore_index=True)

        """This loop is to loop through Ed's data to do the Pearson Correlation"""
        for col_eds in eds_data.columns:

            """Looping through eds_data column to take out the NaN values and the corresponding value in the result 
            column these current columns are a PANDAS series type!"""
            current_column_eds = eds_data[col_eds]
            # current_column_result = result[col_result]

            """If the dataset is LDA then the dropping NaN values and adjusting ranges is not necessary. This is only 
            done for PCA dataset"""
            if dataset_name == "LDA":
                continue

            else:

                """Looping through eds_data column to take out the NaN values and the corresponding value in the result 
                column these current columns are a PANDAS series type!"""
                indices_to_drop = []

                """Need to make a copy of the current column_result to drop NaN values for correlation. I do not want to 
                            drop values for this histograms"""
                current_column = current_column_result.copy()

                for index, value in current_column_eds.items():  # Iterating through the eds_data column
                    if pd.isna(value):  # If there is a null value in the column
                        indices_to_drop.append(index)

                current_column_eds.drop(indices_to_drop, inplace=True)
                current_column.drop(indices_to_drop, inplace=True)

                """Updating the ranges in ranges2 after dropping NaN rows in current_column_pca"""
                for i, (start, end) in enumerate(modified_ranges):
                    adjusted_start = start - sum(index < start for index in indices_to_drop)
                    adjusted_end = end - sum(index <= end for index in indices_to_drop)
                    modified_ranges[i] = (adjusted_start, adjusted_end)

                current_column.reset_index(drop=True, inplace=True)

                """Skipping the rest of the code when there is no correlation (completely empty columns)"""
                if current_column.empty:
                    continue

                """Calculating the correlation that corresponds to the graph"""
                correlation = np.corrcoef(current_column, current_column_eds)[0, 1]
                correlations.loc[f'{dataset_name}_{result.columns.get_loc(col_result) + 1}', col_eds] = correlation

                """Creating a scatter plot with the PCA column values as x values and the eds data column values as the y
                """
                plt.figure()
                plt.scatter(current_column, current_column_eds)
                plt.title(f'Scatter Plot: {dataset_name}_{result.columns.get_loc(col_result) + 1} vs {col_eds}')
                plt.xlabel(f'{dataset_name}_{result.columns.get_loc(col_result) + 1}')
                plt.ylabel(col_eds)
                plt.text(0.6, 0.9, f'Correlation: {correlation: .2f}', transform=plt.gca().transAxes, fontsize=10,
                         color='blue')

                """Making sure that there are no invalid characters in the filename"""
                scatter_filename = f'Scatter_{dataset_name}_{result.columns.get_loc(col_result) + 1}_vs_{col_eds}'
                sanitized_filename_scatter = sanitize_filename(scatter_filename)

                save_plot(plt, sanitized_filename_scatter, dataset_name="Original", folder_path=r"C:\Users\tnt02"
                                                                                                r"\OneDrive"
                                                                                                r"\Desktop\Masters "
                                                                                                r"Research\Running "
                                                                                                r"Data\Last "
                                                                                                r"Run\Correlations")

                plt.close()

    return correlations, all_average_results, all_error_results


def averaged_histogram(result, current_column_result, col_result, input_range, avg_names, current_variance, dev_data,
                       dataset_name):
    """
    Average the data points (PCA components) from the correlation graphs calculate the std and then plot them in a
    histogram with the std. x-axis is polymer name and y-axis is averaged pca value.
    """

    current_column_result = current_column_result.to_frame()  # current_column_pca from a pandas series to a DataFrame
    # print("\nThis is the current column:\n", current_column_result)

    """Add check to see if more than one data point for each polymer to average (error bars)"""
    boolean = check_single_data_point_ranges(input_range)

    if boolean:  # If there is one data point per polymer
        # Just tensile data error
        averaged_points = current_column_result
        error = pd.DataFrame(index=averaged_points.index, columns=averaged_points.columns)
        for index, row in averaged_points.iterrows():
            error.loc[index] = abs(dev_data.iloc[index, :].sum() * row)
        averaged_points = averaged_points.T
        error = error.T
        # print('\nThese are the averaged values:', averaged_points)
        # print('\nThese are the error values:', error)

    else:  # If there is more than one data point per polymer
        if dev_data.empty:  # Just rheology data error
            averaged_pca, avg_names = calculate_avg(current_column_result, input_range, avg_names)
            averaged_points = averaged_pca.iloc[:, [0]].copy()
            error = averaged_pca.iloc[:, [1]].copy()
            averaged_points = averaged_points.T
            error = error.T

            # print('\nThese are the averaged values:', averaged_points)
            # print('\nThese are the error values:', error)

        else:  # Rheology and tensile data error (STILL WORKING ON - NOT CORRECT!!!!)
            averaged_pca, avg_names = calculate_avg(current_column_result, input_range, avg_names)
            averaged_points = averaged_pca.iloc[:, [0]].copy()
            error_rheology = averaged_pca.iloc[:, [1]].copy()
            # print('\nRheology Error: \n', error_rheology)

            # dev_data_sum = dev_data.sum(axis=1)
            # error_tensile = abs(dev_data_sum * averaged_points.iloc[:, 0]).to_frame()
            # print('\nTensile Error: \n', error_tensile)

            """Taking the averaged pca values (averaged_points) and multiplying them by the sum of deviations"""
            error_tensile = pd.DataFrame(index=averaged_points.index, columns=averaged_points.columns)

            for index, row in averaged_points.iterrows():
                error_tensile.loc[index] = abs(dev_data.iloc[index, :].sum() * row)

            # print('\nRheology Error: \n', error_rheology)
            # print('\nTensile Error: \n', error_tensile)

            averaged_points = averaged_points.T

            error = error_rheology.iloc[:, 0] + error_tensile.iloc[:, 0]
            error = error.T
            # print('\nThese are the averaged values:', averaged_points)
            # print('\nThese are the error values:', error)

    """Creating the histogram"""
    fig, ax = plt.subplots()
    ax.bar(range(len(avg_names)), averaged_points.iloc[0],
           yerr=error.iloc[0], capsize=5)
    ax.set_title(f"Histogram: {dataset_name}{result.columns.get_loc(col_result) + 1}", fontsize=16)
    ax.set_ylabel(f"{dataset_name}{result.columns.get_loc(col_result) + 1}", fontsize=12)
    ax.set_xlabel("Polymer", fontsize=12)
    ax.set_xticks(range(len(avg_names)))
    ax.set_xticklabels(avg_names, rotation=90, fontsize=12)
    ax.set_yticklabels(ax.get_yticks(), fontsize=12)
    ax.text(0.6, 1.1, f'Variance Explained: {current_variance: .2f}', transform=ax.transAxes, fontsize=14,
            color='black')

    histogram_filename = f'Histogram_{dataset_name}{result.columns.get_loc(col_result) + 1}'
    sanitized_filename_histogram = sanitize_filename(histogram_filename)

    save_plot(fig, sanitized_filename_histogram, dataset_name="Original", folder_path=r"C:\Users\tnt02"
                                                                                      r"\OneDrive"
                                                                                      r"\Desktop\Masters "
                                                                                      r"Research\Running "
                                                                                      r"Data\Last "
                                                                                      r"Run\Correlations")

    plt.close()

    return averaged_points, error


def distance_radii(polymer_lots, result, ranges, dataset_name):
    averaged_data_lda = []
    cluster_radii = []

    """Calculating the means for each polymer lot"""
    for start, end in ranges:
        block_lda = result.iloc[start:end + 1].values  # Pulling the values to be averaged for one polymer lot
        avg_block_lda = block_lda.mean(axis=0).tolist()
        averaged_data_lda.append(avg_block_lda)
    averaged_data_lda_df = pd.DataFrame(averaged_data_lda)

    """Calculating the distances from lot 30035"""
    datapoint = averaged_data_lda_df.iloc[0]  # Making 30035 reference polymer
    distances = calculate_distances(averaged_data_lda_df, datapoint, distance_metric='euclidean')

    """Calculating the radii for the polymer lot clusters"""
    # Finding distances between centroid and each same polymer lot test
    for start, end in ranges:  # Need a loop to calculate individual cluster distances
        lot_data = result.iloc[start:end + 1].values
        lot_data_df = pd.DataFrame(lot_data)
        cluster_centroid = lot_data.mean(axis=0)  # This becomes the reference point for distances
        cluster_centroid_df = pd.Series(cluster_centroid)
        within_cluster_distances = calculate_distances(lot_data_df, cluster_centroid_df, distance_metric='euclidean')
        radii = within_cluster_distances.std(axis=0)
        cluster_radii.append(radii)

    cluster_radii_pd = pd.DataFrame(cluster_radii)

    polymer_lots_df = pd.DataFrame(polymer_lots)

    """Making a table that has the polymer lot names, distances, and radii"""
    result = pd.concat([polymer_lots_df, distances, cluster_radii_pd], axis=1)
    result.columns = ["Polymer", "Distance from 30035", "Radii"]
    result.sort_values(by="Distance from 30035", inplace=True)

    print("This is the distances before scaling: \n")
    print(result)

    """Scaling the Results to fall within 0 to 1"""
    scaler = MinMaxScaler()
    scaler.fit(result[['Distance from 30035']])
    result[['Distance from 30035']] = scaler.transform(result[['Distance from 30035']])

    # Get the min and max values used by the scaler for 'Distance from 30035'
    distance_min = scaler.data_min_[0]
    distance_max = scaler.data_max_[0]

    print(distance_min)
    print(distance_max)

    # Apply the same scaling to the 'Radii' column using the min and max from 'Distance from 30035'
    result['Radii'] = (result['Radii'] - distance_min) / (distance_max - distance_min)
    print("\nScaled Results:\n")
    print(result)

    """Saving the DataFrame to an Excel file"""
    result.to_excel(r'C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running Data\Last Run\Excel '
                    f'Files\Distance_Radii_Table_{dataset_name}.xlsx', sheet_name='sheet1', index=False)


def stress_strain_plots(sample_name, stress_data, strain_data, dataset_name, ranges):

    # Creating a dictionary to keep track of test number
    sample_counter = {}

    # Lists to hold all stress and strain data for combined plot
    all_stress_data = []
    all_strain_data = []
    all_sample_names = []

    # Need a for loop to iterate over all the tests (rows) in the dataframes
    for index, row in stress_data.iterrows():

        sample_name_index = sample_name[index]

        if sample_name_index not in sample_counter:
            sample_counter[sample_name_index] = 1
        else:
            sample_counter[sample_name_index] += 1

        sample_count = sample_counter[sample_name_index]

        fig_stress_strain, ax_stress_strain = plt.subplots()
        ax_stress_strain.scatter(strain_data.iloc[index, :], stress_data.iloc[index, :])
        ax_stress_strain.set_title(f"Stress-Strain Curve {sample_name[index]}_{sample_count} {dataset_name}")
        ax_stress_strain.set_ylabel('Stress')
        ax_stress_strain.set_xlabel('Strain')
        plt.grid(color='k', linestyle='--', linewidth='0.25')

        """Calling to save function"""
        save_plot(fig_stress_strain, f"Stress_Strain_{sample_name[index]}_{sample_count}.png", dataset_name,
                  os.path.join(
                      r"C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running Data\Last Run\Stress Strain"))

        # Collecting data for combined plot
        all_stress_data.append(stress_data.iloc[index, :])
        all_strain_data.append(strain_data.iloc[index, :])
        all_sample_names.append(f"{sample_name[index]}_{sample_count}")

        plt.close()

    if dataset_name == "Interpolated":

        # Combined plot
        fig_combined, ax_combined = plt.subplots()

        # Defining a list of colors and markers for plotting
        # colors = plt.cm.get_cmap('tab10', len(set(all_sample_names)))
        custom_colors = ['#1F77B4',  '#FF7F0E', '#2CA02C', '#9467BD', '#8C564B', '#BCBD22', '#7F7F7F']
        markers = ['o', 's', '^', 'v', 'D', '>', 'p']

        # For tracking used colors and markers
        color_map = {}
        marker_map = {}

        # Extracting unique sample groups
        sample_groups = sorted(set(name.split('_')[0] for name in all_sample_names))
        test_numbers = sorted(set(name.split('_')[1] for name in all_sample_names))

        print(sample_groups)

        # Assigning colors and markers to each unique sample group
        for i, group in enumerate(sample_groups):
            color_map[group] = custom_colors[i % len(sample_groups)]

        for i, test_number in enumerate(test_numbers):
            marker_map[test_number] = markers[i % len(markers)]

        # Creating separate plots for each sample group
        for group in sample_groups:
            fig, ax = plt.subplots(figsize=(8, 6))

            ax.set_title(f'Stress-Strain Curves for {group}')
            ax.set_xlabel('Strain')
            ax.set_ylabel('Stress')
            ax.grid(color='k', linestyle='--', linewidth=0.5)

            for strain, stress, name in zip(all_strain_data, all_stress_data, all_sample_names):
                if group == name.split('_')[0]:
                    test_number = name.split('_')[1]
                    ax.plot(strain, stress, label=f'{group}_{test_number}', color=color_map[group],
                            marker=marker_map[test_number], markersize=5, linewidth=1.5)

            ax.legend(loc="upper right", fontsize='small')

            save_plot(fig, f"Combined Stress Strain Plots {group}.png", dataset_name,
                      os.path.join(
                          r"C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running Data\Last Run\Stress Strain"))

        # Creating the combined plot
        for strain, stress, name in zip(all_strain_data, all_stress_data, all_sample_names):

            group, test_number = name.split('_')

            # Plot stress-strain data with appropriate color and linestyle
            ax_combined.plot(strain, stress, label=name, color=color_map[group], marker=marker_map[test_number], markersize=3, linewidth=0.5)

        ax_combined.set_title('Combined Stress-Strain Curves')
        ax_combined.set_ylabel('Stress')
        ax_combined.set_xlabel('Strain')
        plt.grid(color='k', linestyle='--', linewidth=0.25)

        ax_combined.legend(loc="lower center", bbox_to_anchor=(0.5, -0.7), fontsize='small', ncol=5)

        save_plot(fig_combined, f"Interpolated Stress Strain Plots.png", dataset_name,
                  os.path.join(
                      r"C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running Data\Last Run\Stress Strain"))

        plt.close()


def preprocessing_tensile_data(force_data, stress_data, strain_data):
    force_data.fillna(0, inplace=True)
    stress_data.fillna(0, inplace=True)

    """Removing data w/ force less than 0 and replacing NaN values with 0
    setting strain for the first force = 0 and shifting data"""
    # Looping through rows
    for index, row in force_data.iterrows():

        positive_found = False
        negative_found = False

        # Looping through items in row
        for column, value in row.items():

            # Checking for the first positive value.
            if value > 0:
                positive_found = True

            # Getting where the positive values become negative again because this is where the data needs to be
            # truncated
            if positive_found and value < 0:
                negative_found = True

            # Replacing the negative values with NaN and also truncating the values after the first negative after
            # positive values
            if negative_found or value < 0:
                # Set negative value to NaN
                force_data.at[index, column] = None
                stress_data.at[index, column] = None
                strain_data.at[index, column] = None

    # Apply the function to each row
    force_data = force_data.apply(shift_row, axis=1)
    stress_data = stress_data.apply(shift_row, axis=1)
    strain_data = strain_data.apply(shift_row, axis=1)

    force_data.fillna(0, inplace=True)
    stress_data.fillna(0, inplace=True)

    # Drop columns with all zeros from data
    force_data = force_data.loc[:, (force_data != 0).any(axis=0)]
    stress_data = stress_data.loc[:, (stress_data != 0).any(axis=0)]
    strain_data = strain_data.loc[:, (strain_data != 0).any(axis=0)]

    strain_data = strain_data.dropna(axis=1, how='all')

    print("Strain before extrapolation: \n")
    print(strain_data)

    longest_elongation = None
    longest_row_length = 0

    # This is a loop to extrapolate the data of each row as to fill in the NaN values
    for index, row in strain_data.iterrows():

        # This is a loop to find the longest elongation in the entire dataframe
        for index2, row2 in strain_data.iterrows():
            if len(row2.dropna()) > longest_row_length:
                longest_row_length = len(row2.dropna())

                # Checking if the longest elongation needs to be updated
                if longest_elongation is None or row2.dropna().iloc[-1] > longest_elongation:
                    longest_elongation = row2.dropna().iloc[-1]
                    # print("Longest Elongation:")
                    # print(longest_elongation)

        # Take the last value in the row and subtract it from the longest elongation in the dataframe
        row_longest_elongation = row.dropna().iloc[-1]
        row_longest_elongation_index = len(row.dropna())
        elongation_difference = longest_elongation - row_longest_elongation
        # Take the index of the longest_row_length and subtract the index for the last value of the row
        index_difference = longest_row_length - row_longest_elongation_index
        # Take the last value-longest_elongation and divide the value by the difference of index
        incremental_step = elongation_difference / index_difference

        # For the rows with NaN values add the value to the last value for the next element in the row
        for col in row.index:
            last_value = row.dropna().iloc[-1]
            if pd.isnull(row[col]):
                if col >= longest_row_length:
                    break
                row[col] = last_value + incremental_step
                last_value += incremental_step

    print("Strain after extrapolation: \n")
    print(strain_data)

    # Shifting the strain data so the first force value strain is 0
    # Subtract the first value in each row from the entire ro
    strain_data = strain_data.sub(strain_data.iloc[:, 0], axis=0)

    return stress_data, strain_data


def interpolation(stress_data, strain_data):
    # Creating new dataframes to put the interpolated data in
    interpolated_stress = []
    interpolated_strain = []

    loop_index = 1

    for index, row in strain_data.iterrows():
        loop_index = loop_index + 1

        strain_row = row.values
        stress_row = stress_data.loc[index].values

        min_strain = strain_row.min()
        max_strain = strain_row.max()

        # Define the range of strain values for interpolation
        interpolated_strain_values_row = np.linspace(min_strain, max_strain, 600)
        # Applying linear interpolation
        interpolated_stress_values_row = np.interp(interpolated_strain_values_row, strain_row, stress_row)

        # Storing interpolated strain and stress values in the new DataFrames
        interpolated_strain.append(interpolated_strain_values_row)
        interpolated_stress.append(interpolated_stress_values_row)

    interpolated_strain_df = pd.DataFrame(interpolated_strain)
    interpolated_stress_df = pd.DataFrame(interpolated_stress)

    print("Interpolated Strain Values: \n")
    print(interpolated_strain_df)
    print("Interpolated Stress Values: \n")
    print(interpolated_stress_df)

    return interpolated_strain_df, interpolated_stress_df


def shift_row(row):
    first_valid_index = row.first_valid_index()
    if first_valid_index is not None:
        num_shift = row.index.get_loc(first_valid_index)
        return row.shift(-num_shift)
    return row


def check_single_data_point_ranges(ranges):
    """
    This function is used to check the range of the dataset to see if the data has already been averaged.
    """

    for start, end in ranges:
        if start != end:
            return False
    return True


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


def save_tables(table_data, name_cluster_data, dataset_name, folder_path=""):
    """
    This function creates and saves the clustering result tables to Excel files
    """

    """Creating specified filenames for different tables"""
    pca_components_filename = f'{dataset_name}_components_table.xlsx'
    pca_cluster_filename = f'{dataset_name}_cluster_table.xlsx'

    """Printing the clustered data to an Excel file"""
    name_cluster_data.to_excel(os.path.join(folder_path, pca_cluster_filename), sheet_name='sheet1', index=False)
    print("\nTable with all Component Cluster Assignments:")
    print(name_cluster_data.to_string(index=False))

    """Creating DataFrame from table_data"""
    table_df = pd.DataFrame(table_data)
    print("\nTable with Components, Optimized Clusters, and Variance Explained:\n")
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

