"""Importing packages"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import os
import re

from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
from sklearn.metrics import pairwise_kernels
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean, cityblock, cosine
from sklearn.decomposition import KernelPCA
from pathlib import Path

"""This is for input files"""
path_to_input_rhe = Path(__file__).parent / 'Excel Files' / 'Rheology'
path_to_input_ten = Path(__file__).parent / 'Excel Files' / 'Tensile'


def making_output_folders():
    path_to_output = Path(__file__).parent / 'Outputs'  # This is for outputs
    path_to_output.mkdir(exist_ok=True)  # Making output folder

    """Creating subfolders within the output folder"""
    subfolders = ['Stress Strain Plots', 'Rheology Plots', 'Dim Weights', 'Per Var-Dis', 'Comparing Ed Data',
                  'Clustering Opt', 'Excel Files']
    for folder in subfolders:
        (path_to_output / folder).mkdir(exist_ok=True)  # Making subfolders

        """Making subsubfolders for the different clustering optimization techniques"""
        if folder == 'Clustering Opt':
            (path_to_output / folder / 'Elbow Method').mkdir(exist_ok=True)
            (path_to_output / folder / 'Silhouette Scores').mkdir(exist_ok=True)
        elif folder == 'Excel Files':
            (path_to_output / folder / 'PCA').mkdir(exist_ok=True)
            (path_to_output / folder / 'LDA').mkdir(exist_ok=True)
        elif folder == 'Stress Strain Plots':
            (path_to_output / folder / 'Group Polymer Lot').mkdir(exist_ok=True)
            (path_to_output / folder / 'Ind Polymer Lot').mkdir(exist_ok=True)



    print("")
    print("This is the file path for the outputs folder:", path_to_output)

    return path_to_output


"""Here are the loading functions that are used by the different mains."""


def load_embeddings_rheology(raw_data, dataset_chosen):
    """
        Purpose: This function takes in an Excel file with multiple sheets for rheology testing data - we only took
        viscosity, storage, and loss moduli data from the Excel sheets. This function also standardized the dataset.

        Input:
            1) PANDAS dataframes: raw_data

        Output:
            1) PANDAS dataframes: concatenated_data_df, frequency_data, eds_data,
            2) Lists: sample_names
            3) Ints: num_datapoints
        """

    """Creating an ExcelFile object to read the Excel file"""
    xls = pd.ExcelFile(raw_data)

    """Getting a list of all the sheet names (multiple sheets in the excel file)"""
    sheet_names = xls.sheet_names

    """Creating empty lists to store individual data DataFrames and sample names"""
    sample_names = []
    concatenated_rows = []
    frequencies_rows = []

    """Checking if you want to log the dataset"""
    logging = str(input('\n\nDo you want to log the data (y or n)?\n'))

    """Looping through each sheet and reading data into a dataframe"""
    for index, sheet_name in enumerate(sheet_names):

        if index == 0:  # Skip the first sheet
            print("Ed's Calculations: ")
            eds_data = pd.read_excel(xls, sheet_name=sheet_name, usecols="C:E")
            print(eds_data)
            continue

        """Reading in the data from the current sheet into a DataFrame we want only values pertaining to columns D
        through G since indexing starts at 0 column D is index 3 and so on"""
        df = pd.read_excel(xls, sheet_name=sheet_name, usecols="D:G")

        """Here is where I truncated the dataset manually for the truncation analysis - change these values if you need
        different truncations"""
        if dataset_chosen == '2055-2060 Series 70C Batch 2 Outliers.xlsx':
            df = df.iloc[0: 13, :]  # 70C Batch 2 only collected 13 datapoints for each polymer

        """Placing all the values for frequency, storage modulus (G'), loss modulus (G''), and viscosity in their own
        lists so that they can be added into the DataFrames"""
        frequencies_row = df.iloc[:, 0].tolist()
        g_prime = df.iloc[:, 1].tolist()
        g_double_prime = df.iloc[:, 2].tolist()
        viscosity = df.iloc[:, 3].tolist()

        """Logging all the data"""
        if logging == 'y':
            g_prime = [np.log10(value) for value in g_prime]
            g_double_prime = [np.log10(value) for value in g_double_prime]
            viscosity = [np.log10(value) for value in viscosity]

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

    """Applying StandardScaler to prevent bias"""
    scaler = StandardScaler()
    concatenated_data_scaled = scaler.fit_transform(concatenated_data)
    concatenated_data_df = pd.DataFrame(concatenated_data_scaled, columns=concatenated_data.columns)

    """Print statements for check"""
    # print("Data After Standardizing:")
    # print(concatenated_data_df)
    # print("")

    frequency_data = pd.DataFrame(frequencies_rows)

    num_datapoints = len(g_prime)

    return sample_names, concatenated_data_df, frequency_data, eds_data, num_datapoints


def load_embeddings_tensile(raw_data):
    """
    Purpose: This function takes in an Excel file with multiple sheets for tensile testing data. When data is read in a
    sheet is specified for a certain temperature. This function can standardize the data.

    Input:
        1) PANDAS dataframes: raw_data

    Output:
        1) PANDAS dataframes: stress_data, strain_data, force_data, eds_data
        2) Lists: sample_names, original_ranges
        3) Ints: num_datapoints
    """

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

    """Applying StandardScaler"""
    # scaler = StandardScaler()
    #  stress_data_scaled = scaler.fit_transform(stress_data)
    # stress_data = pd.DataFrame(stress_data_scaled, columns=stress_data.columns)

    """Print Check"""
    # print("Data After Standardizing:")
    # print(stress_data)
    # print("")

    return sample_names, stress_data, strain_data, force_data, num_datapoints, eds_data


"""This type of dataset was quickly abandoned therefore no mains were included for this"""


def load_embeddings_combined():
    """
    Purpose: This function loads in Excel data for datasets containing rheological and tensile testing data. The tensile
     data was added to the dataset at the end of each row for each polymer lot test. (This was an avenue we did not
     elaborate on and will likely not be of any use)

    Input: N/A

    Output:
        1) PANDAS dataframes: concatenated_data_df, frequency_data, eds_data
        2) Lists: sample_names
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


"""Additional tensile dataset preprocessing functions."""


def preprocessing_tensile_data(force_data, stress_data, strain_data):
    """
    Purpose: This function preprocesses tensile data to prevent any errors due to inaccuracies or problematic data such
    as NaN values, etc.

    Input:
        1) PANDAS dataframes: force, stress, and strain data

    Output: Preprocessed PANDAS dataframes: stress and strain data that:
        1) do not contain negative values
        2) contains the same number of dimensions
        3) shifted so the first value in each test is zero.
    """

    force_data.fillna(0, inplace=True)
    stress_data.fillna(0, inplace=True)

    """Removing data w/ force less than 0 and replacing NaN values with 0
    setting strain for the first force = 0 and shifting data"""

    """Looping through rows"""
    for index, row in force_data.iterrows():

        positive_found = False
        negative_found = False

        """Looping through items in row"""
        for column, value in row.items():

            """Checking for the first positive value."""
            if value > 0:
                positive_found = True

            """Getting where the positive values become negative again because this is where the data needs to be
            truncated"""
            if positive_found and value < 0:
                negative_found = True

            """Replacing the negative values with NaN and also truncating the values after the first negative after
            positive values"""
            if negative_found or value < 0:
                """Setting negative value to NaN"""
                force_data.at[index, column] = None
                stress_data.at[index, column] = None
                strain_data.at[index, column] = None

    """Applying the function to each row"""
    force_data = force_data.apply(shift_row, axis=1)
    stress_data = stress_data.apply(shift_row, axis=1)
    strain_data = strain_data.apply(shift_row, axis=1)

    force_data.fillna(0, inplace=True)
    stress_data.fillna(0, inplace=True)

    """Dropping columns with all zeros from data"""
    # force_data = force_data.loc[:, (force_data != 0).any(axis=0)]
    stress_data = stress_data.loc[:, (stress_data != 0).any(axis=0)]
    strain_data = strain_data.loc[:, (strain_data != 0).any(axis=0)]

    strain_data = strain_data.dropna(axis=1, how='all')

    """Print Check"""
    # print("Strain before extrapolation: \n")
    # print(strain_data)

    longest_elongation = None
    longest_row_length = 0

    """This is a loop to extrapolate the data of each row as to fill in the NaN values"""
    for index, row in strain_data.iterrows():

        """This is a loop to find the longest elongation in the entire dataframe"""
        for index2, row2 in strain_data.iterrows():
            if len(row2.dropna()) > longest_row_length:
                longest_row_length = len(row2.dropna())

                """Checking if the longest elongation needs to be updated"""
                if longest_elongation is None or row2.dropna().iloc[-1] > longest_elongation:
                    longest_elongation = row2.dropna().iloc[-1]

                    """Print check"""
                    # print("Longest Elongation:")
                    # print(longest_elongation)

        """Taking the last value in the row and subtracting it from the longest elongation in the dataframe"""
        row_longest_elongation = row.dropna().iloc[-1]
        row_longest_elongation_index = len(row.dropna())
        elongation_difference = longest_elongation - row_longest_elongation

        """Taking the index of the longest_row_length and subtracting the index for the last value of the row"""
        index_difference = longest_row_length - row_longest_elongation_index

        """Taking the last value-longest_elongation and dividing the value by the difference of index"""
        incremental_step = elongation_difference / index_difference

        """For the rows with NaN values adding the value to the last value for the next element in the row"""
        for col in row.index:
            last_value = row.dropna().iloc[-1]
            if pd.isnull(row[col]):
                if col >= longest_row_length:
                    break
                row[col] = last_value + incremental_step
                last_value += incremental_step

    """Print Check"""
    # print("Strain after extrapolation: \n")
    # print(strain_data)

    """Shifting the strain data so the first force value strain is 0 by subtracting the first value in each row from the
     entire row"""
    strain_data = strain_data.sub(strain_data.iloc[:, 0], axis=0)

    return stress_data, strain_data


def interpolation(stress_data, strain_data):
    """
    Purpose: This function interpolates the tensile data. Through preprocessing we have the same number of points however each
    strain sampling is different. To effectively compare we need to ensure that we are comparing stresses that
    correspond to the same strain values.

    Inputs:
        1) PANDAS dataframes: stress_data and strain_data (preprocessed)

    Outputs:
        1) PANDAS dataframes: stress_data and strain_data (preprocessed and interpolated)
    """

    """Asking user for input"""
    num_datapoints = int(input('\n\nHow many datapoints would you like to use for interpolation?'))

    """Creating new dataframes to put the interpolated data in"""
    interpolated_stress = []
    interpolated_strain = []

    """Index counter"""
    loop_index = 1

    """Looping through strain data to create evenly spaced strain values for each test"""
    for index, row in strain_data.iterrows():
        loop_index = loop_index + 1

        strain_row = row.values
        stress_row = stress_data.loc[index].values

        min_strain = strain_row.min()
        max_strain = strain_row.max()

        """Defining the range of strain values for interpolation (The 600 value indicated 600 interpolated data points -
        this can be changed to whatever number you want to use.)"""
        interpolated_strain_values_row = np.linspace(min_strain, max_strain, num_datapoints)

        """Applying linear interpolation"""
        interpolated_stress_values_row = np.interp(interpolated_strain_values_row, strain_row, stress_row)

        """Storing interpolated strain and stress values in the new DataFrames"""
        interpolated_strain.append(interpolated_strain_values_row)
        interpolated_stress.append(interpolated_stress_values_row)

    interpolated_strain_df = pd.DataFrame(interpolated_strain)
    interpolated_stress_df = pd.DataFrame(interpolated_stress)

    """Print Check"""
    # print("Interpolated Strain Values: \n")
    # print(interpolated_strain_df)
    # print("Interpolated Stress Values: \n")
    # print(interpolated_stress_df)

    return interpolated_strain_df, interpolated_stress_df


"""These functions create graphs to interpret the original data within the various datasets. The rheology datasets plot
the viscosity, and storage and loss moduli data vs frequency while the tensile datasets plot stress vs strain. These 
graphs help us to gain insights into the general trends of the dataset, standard deviations between the same material
tests, obvious outliers, etc."""


def initial_data_graphs_rheology(data, frequency_data, num_datapoints, path_to_output):
    """
    Purpose: This function is only used for rheology data to produce graphs of each of the features in relation to
    frequency. (3 subplots)

    Input:
        1) PANDAS Dataframes: data, frequency_data
        2) Strs: dataset_name
        3) Ints: num_datapoints

    Output: N/A
    """

    """Defining data and labels"""
    data_sets = ["G' [Pa]", "G'' [Pa]", "|n*| [Pa]"]

    """THIS CHANGES DEPENDING ON THE DATASET - I would definitely change this if in future iterations to be more dynamic
    and not hard coded. - I have it all commented out because I can't remember which dataset this is for and if not 
    correct it throws an error."""
    # legend_labels = ["2055_.1", "2055_.2", "2055_.3", "2055_.4", "2055_.5", "2055_.6", "2055_.7", "2056_.1", "2056_.2",
    #                 "2056_.3", "2056_.4", "2056_.5", "2056_.6", "2056_.7", "2057_.1", "2057_.2", "2057_.3", "2057_.5",
    #                 "2057_.6", "2057_.7", "2058_.1", "2058_.2", "2058_.3", "2058_.4", "2058_.5", "2058_.6", "2058_.7",
    #                 "2059_.1", "2059_.2", "2059_.3", "2059_.4", "2059_.5", "2059_.6", "2059_.7", "2060_.1", "2060_.2",
    #                 "2060_.3", "2060_.4", "2060_.5", "2060_.6", "2060_.7", "30035_.1", "30035_.2", "30035_.3",
    #                 "30035_.4", "30035_.5", "30035_.6", "30035_.7"]

    freq1 = frequency_data.iloc[0][0:num_datapoints].tolist()

    """Creating the figure"""
    fig_rheology, axs = plt.subplots(3, sharex=True, figsize=(10, 6))
    """Adjusting the vertical spacing"""
    plt.subplots_adjust(hspace=0.5, right=0.8)

    """Commented out one is for single polymer"""
    # line_styles = ['-', ':', '--', '']
    line_styles = ['-', ':', '--', '-.', '-', ':', '--']

    """Generating the individual subplots"""
    for i, ax in enumerate(axs):
        ax.set_title(f"{data_sets[i]} VS Freq")
        ax.set_ylabel(data_sets[i])

        if i == 2:
            ax.set_xlabel('Frequency [Hz]')

        for j in range(num_datapoints):
            data_values = data.iloc[j][i * num_datapoints:(i + 1) * num_datapoints].tolist()
            """Using different line styles for every 7 plots"""
            line_style = line_styles[j // 7]
            """Assigning a different color for every seventh line"""
            color = f'C{j % 7}'
            ax.plot(freq1, data_values, linestyle=line_style, color=color)

            """Commented out because of legend"""
            # ax.plot(freq1, data_values, label=legend_labels[j], linestyle=line_style, color=color)

    # ax.legend(legend_labels, loc="center left", bbox_to_anchor=(1, 3.2), fontsize='small', ncol=2) # single polymer
    # ax.legend(legend_labels, loc="upper left", bbox_to_anchor=(1, 3.2), fontsize='small', ncol=2) # multi polymer

    fig_rheology.savefig(path_to_output / 'Rheology Plots' / f"Standardized Data.png")


def initial_data_graphs_high_res_rheology(data, frequency_data, dataset_name, num_datapoints, ranges, path_to_output):
    """
    Purpose: This function is only used for the high-res rheology data to produce graphs of each of the features in
    relation to frequency. (3 subplots)

    Input:
        1) PANDAS Dataframes: data, frequency_data
        2) Strs: dataset_name
        3) Ints: num_datapoints
        4) Lists: ranges

    Output: N/A
    """

    """Defining data and labels"""
    data_sets = ["G' [Pa]", "G'' [Pa]", "|n*| [Pa]"]

    """THIS CHANGES DEPENDING ON THE DATASET - I would definitely change this if in future iterations to be more dynamic
     and not hard coded. - If not correct it throws an error."""
    legend_labels = ["30035_.1", "30035_.2", "30035_.3", "30035_.4", "30035_.5", "30035_.6", "30035_.7",
                     "2055_.4", "2055_.5", "2055_.6", "2055_.7",
                     "2056_.1", "2056_.2", "2056_.3", "2056_.4", "2056_.5", "2056_.6", "2056_.7",
                     "2057_.1", "2057_.2", "2057_.3", "2057_.4", "2057_.5", "2057_.6", "2057_.7",
                     "2058_.1", "2058_.2", "2058_.3", "2058_.4", "2058_.5", "2058_.6", "2058_.7",
                     "2059_.1", "2059_.2", "2059_.3", "2059_.4", "2059_.5", "2059_.6", "2059_.7",
                     "2060_.1", "2060_.2", "2060_.3", "2060_.4", "2060_.5", "2060_.6", "2060_.7", ]

    freq1 = frequency_data.iloc[0][0:num_datapoints].tolist()

    """Creating plot"""
    fig, axs = plt.subplots(3, sharex=True, figsize=(10, 6))

    """Adjusting the vertical spacing"""
    plt.subplots_adjust(hspace=0.5, right=0.8)

    """Commented out one is for single polymer"""
    marker_styles = ['o', 's', '^', 'v', 'D', '>', 'p']
    # colors = ["black", "orange", "blue", "pink", "purple", "red", "yellow"]
    colors = ['#BCBD22', '#1F77B4', '#FF7F0E', '#2CA02C', '#9467BD', '#8C564B', '#7F7F7F']
    polymer_colors = []
    test_styles = ['o', 's', '^', 'v', 'D', '>', 'p', 'v', 'D', '>', 'p']  # Assigned the line styles for 2055 since
    # doesn't start at 1

    """Generating the individual subplots"""
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
                    marker=test_styles[j], markersize=5)

    # ax.legend(legend_labels, loc="lower left", bbox_to_anchor=(1, 3.2), fontsize='small', ncol=2) # single polymer
    ax.legend(legend_labels, loc="upper left", bbox_to_anchor=(1, 3.2), fontsize='small', ncol=2) # mult polymer

    fig.savefig(path_to_output / 'Rheology Plots' / f"Standardized_High_Prec_Data.png")


def stress_strain_plots(sample_name, stress_data, strain_data, dataset_name, ranges, path_to_output):
    """
    Purpose: This function is only used for the tensile data to produce stress vs strain graphs. It produces
    stress/strain plots for the individual polymer lot tests and a combined stress_strain plot for all tests belonging
    to the same polymer lot.

    Input:
        1) PANDAS Dataframes: stress_data, strain_data
        2) Strs: dataset_name
        4) Lists: sample_name, ranges

    Output: N/A
    """

    """Creating a dictionary to keep track of test number"""
    sample_counter = {}

    """Lists to hold all stress and strain data for combined plot"""
    all_stress_data = []
    all_strain_data = []
    all_sample_names = []

    """Need a for loop to iterate over all the tests (rows) in the dataframes"""
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

        fig_stress_strain.savefig(path_to_output / 'Stress Strain Plots' / 'Ind Polymer Lot' / f"Stress_Strain_{sample_name[index]}_{sample_count}.png")

        """Collecting data for combined plot"""
        all_stress_data.append(stress_data.iloc[index, :])
        all_strain_data.append(strain_data.iloc[index, :])
        all_sample_names.append(f"{sample_name[index]}_{sample_count}")

        plt.close()

    if dataset_name == "Interpolated":

        """Combined plot"""
        fig_combined, ax_combined = plt.subplots()

        """Defining a list of colors and markers for plotting"""
        # colors = plt.cm.get_cmap('tab10', len(set(all_sample_names)))
        custom_colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#9467BD', '#8C564B', '#BCBD22', '#7F7F7F']
        markers = ['o', 's', '^', 'v', 'D', '>', 'p']

        """For tracking used colors and markers"""
        color_map = {}
        marker_map = {}

        """Extracting unique sample groups"""
        sample_groups = sorted(set(name.split('_')[0] for name in all_sample_names))
        test_numbers = sorted(set(name.split('_')[1] for name in all_sample_names))

        """Assigning colors and markers to each unique sample group"""
        for i, group in enumerate(sample_groups):
            color_map[group] = custom_colors[i % len(sample_groups)]

        for i, test_number in enumerate(test_numbers):
            marker_map[test_number] = markers[i % len(markers)]

        """Creating separate plots for each sample group"""
        for group in sample_groups:
            fig_lot_grouped, ax = plt.subplots(figsize=(8, 6))

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

            fig_lot_grouped.savefig(
                path_to_output / 'Stress Strain Plots' / 'Group Polymer Lot' / f"Combined Stress Strain Plots {group}.png")

        """Creating the combined plot"""
        for strain, stress, name in zip(all_strain_data, all_stress_data, all_sample_names):
            group, test_number = name.split('_')

            """Plot stress-strain data with appropriate color and linestyle"""
            ax_combined.plot(strain, stress, label=name, color=color_map[group], marker=marker_map[test_number],
                             markersize=3, linewidth=0.5)

        ax_combined.set_title('Combined Stress-Strain Curves')
        ax_combined.set_ylabel('Stress')
        ax_combined.set_xlabel('Strain')
        plt.grid(color='k', linestyle='--', linewidth=0.25)

        ax_combined.legend(loc="lower center", bbox_to_anchor=(0.5, -0.7), fontsize='small', ncol=5)
        # ax_combined.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize='small', ncol=3)

        fig_combined.savefig(
            path_to_output / 'Stress Strain Plots' / "Interpolated Stress Strain Plots.png")

        plt.close()


"""Dimension Reduction Functions"""


def pca_embeddings(data, dataset_name, path_to_output, num_components):
    """
    Purpose: This function applies principal component analysis (PCA) to the dataset.

    Input:
        1) PANDAS Dataframes: data (scaled)
        2) Ints: num_components
        3) Strs: dataset_name

    Output:
        1) result: This is a dataframe that contains the principal component weights (These are the
        components/coefficients for the eigenvectors)
        2) cumulative variance: This is a list containing the percentage (in decimal form) of the total trends captured
        within each of the PCs - these are technically the eigenvalues corresponding to the eigenvectors. The larger
        the eigenvalue the more trends are captured with the corresponding eigenvector. The eigenvector with the
        largest associated eigenvalue becomes the first principal component (eigenvectors are ranked through the
        magnitude of their eigenvalues).
        3) This function creates/saves histogram plots for the first 5 eigenvectors showing:
            - The PCA weights
            - Cumulative variance
    """

    """This is where PCA is actually applied to the dataset"""
    pca = PCA(n_components=num_components)  # Defining how many PCs to use
    result = pca.fit_transform(data)  # Transforming the data to the specified PCs to use

    num_features = data.shape[1]

    results = pd.DataFrame(result)
    results.to_excel(path_to_output / 'Excel Files' / f'{dataset_name}' / 'Eigenvectors_PCA_embeddings.xlsx',
                     sheet_name='sheet1', index=False)

    """Checking if there's only one component (PC) because then we do not use subplots. If there is more than one 
    component then subplots have to be made so that we can see how each feature in the component (PC) was weighted"""
    if num_components == 1:

        """Creating the histogram"""
        fig_weights, ax = plt.subplots()
        ax.bar(range(num_features), pca.components_[0], tick_label=data.columns)
        ax.set_title("PCA 1 Component Analysis")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Weight")
        plt.xticks(rotation=90)

    else:

        """Creating subplots for PCA component analysis"""
        fig_weights, axes = plt.subplots(num_components, 1, sharex=True)

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
        fig_weights.savefig(path_to_output / 'Dim Weights' / f"Weights_{num_components}_{dataset_name}.png")

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
        fig_percent_variance.savefig(
            path_to_output / 'Per Var-Dis' / f"Percent_Variance_{num_components}_{dataset_name}.png")

    return result, cumulative_variance


def lda_embeddings(data, sample_names, dataset_name, path_to_output, num_components):
    """
    Purpose: This function applies linear discriminate analysis (LDA) to the dataset. This dimension reduction technique
    differs from PCA because it utilizes polymer lot names (classification). These labels or classes allow the algorithm
    to 1) maximize variance/distance between data belonging different classes and 2) minimize variance/distance between
    data belonging to the same polymer lots. Because we use classes the variance for this method differs slightly. In
    PCA dimension reduction is not limited other than by the datasets original size/dimensionality. LDA is limited by
    the number of classes. This means that PCA can reach 100% total variance when using the original dimensionality of
    the dataset. LDA will never technically reach 100% because it HAS to lose some information in the dataset based on
    the number of classes.

    Input:
        1) PANDAS Dataframes: data (scaled)
        2) Ints: num_components
        3) Strs: dataset_name

    Output:
        1) result: This is a dataframe that contains the component weights (These are the
        components/coefficients for the eigenvectors)
        2) cumulative variance: This is a list containing the percentage (in decimal form) of the total trends captured
        within each of the components - these are technically the eigenvalues corresponding to the eigenvectors.
        3) This function creates/saves histogram plots for the first 5 eigenvectors showing:
            - The LDA weights
            - Cumulative variance
    """

    """Print Check"""
    # print("Here is the lda_embeddings data: \n", data)
    # print("Here are the lda_embeddings sample names: \n", sample_names)

    """This is where LDA is actually applied to the dataset"""
    lda = LinearDiscriminantAnalysis(n_components=num_components)  # Defining how many components
    result = lda.fit_transform(data, sample_names)  # Transforming the data

    """Print Check"""
    # print(result.T)

    num_features = data.shape[1]

    """Print Check"""
    # print('\nThis is LDA Result:\n', result)
    # print('\nThis is LDA Coef: \n', lda.coef_[0])

    lda_mean_embeddings = lda.means_
    lda_mean_embeddings_df = pd.DataFrame(lda_mean_embeddings)

    """Print Check"""
    # print("Here is the mean for lda embeddings:", lda_mean_embeddings)

    lda_mean_embeddings_df.to_excel(path_to_output / 'Excel Files' / f'{dataset_name}' / 'Mean_Vectors_LDA_embeddings.xlsx',
                                    sheet_name='sheet1', index=False)

    eigenvectors = lda.scalings_

    """Print Check"""
    # print(eigenvectors.shape)
    eigenvalues = lda.explained_variance_ratio_

    """Need to transpose eigenvectors should be a 6x42 not 42x6"""
    eigenvectors = eigenvectors.T

    """Print Check"""
    # print("Eigenvalues:", eigenvalues)
    # print("Eigenvectors:", eigenvectors)
    # print(eigenvectors.shape)

    eigenvectors_df = pd.DataFrame(eigenvectors)
    eigenvectors_df.to_excel(path_to_output / 'Excel Files' / f'{dataset_name}' / 'Eigenvectors_LDA_embeddings.xlsx',
                             sheet_name='sheet1', index=False)

    """Print Check"""
    # data_2 = np.array(data)
    # first_point = np.matmul(data_2, eigenvectors.T)
    # print(first_point[:, 0:2])
    # print(first_point.shape)

    """Checking if there's only one component (PC) because then we do not use subplots. If there is more than one 
        component then subplots have to be made so that we can see how each feature in the component (PC) was weighted"""
    if num_components == 1:
        """Creating the histogram"""
        fig_weights, ax = plt.subplots()
        ax.bar(range(num_features), eigenvectors[0], tick_label=data.columns)
        ax.set_title("LDA 1 Component Analysis")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Weight")
        plt.xticks(rotation=90)

    else:
        """Creating subplots for PCA component analysis"""
        fig_weights, axes = plt.subplots(num_components, 1, sharex=True)

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
        fig_weights.savefig(path_to_output / 'Dim Weights' / f"Weights_{num_components}_{dataset_name}.png")

    """Calculating variance ratios"""
    variance = lda.explained_variance_ratio_

    """Print Check"""
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
        fig_percent_variance.savefig(
            path_to_output / 'Per Var-Dis' / f"Percent_Variance_{num_components}_{dataset_name}.png")

    return result, cumulative_variance


"""Clustering Functions"""


def kmean_hyper_param_tuning(data, num_components, dataset_name, path_to_output):
    """
    Purpose: Silhouette Scores and Elbow Method are applied to the dimension reduction results to determine the optimum
     number of clusters to use for the dataset. Histograms and plots are created for visual aid.

    Input:
        1) PANDAS Dataframes: data
        2) Ints: num_components
        3) Strs: dataset_name

    Output:
        1) Ints: best_grid['n_clusters']
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

        """Print Check"""
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
        ax_silhouette.bar(range(len(silhouette_scores)), list(silhouette_scores), align='center',
                          width=0.5, color='paleturquoise')
        ax_silhouette.set_xticks(range(len(silhouette_scores)))
        ax_silhouette.set_xticklabels(list(parameters))
        # ax_silhouette.set_title('Silhouette Score', fontweight='bold')
        ax_silhouette.set_xlabel('Number of Clusters')
        ax_silhouette.set_ylabel('Silhouette Score')

        fig_silhouette.savefig(
            path_to_output / 'Clustering Opt' / 'Silhouette Scores' / f"{num_components}_numcomp_{dataset_name}.png")

        """Plotting the elbow graph"""
        fig_elbow, ax_elbow = plt.subplots()
        ax_elbow.plot(parameters, inertias, marker='o', linestyle='-')
        ax_elbow.set_title('Elbow Method for Optimal k')
        ax_elbow.set_xlabel('Number of Clusters (k)')
        ax_elbow.set_ylabel('Inertia')
        ax_elbow.set_xticks(parameters)

        fig_elbow.savefig(
            path_to_output / 'Clustering Opt' / 'Elbow Method' / f"{num_components}_numcomp_{dataset_name}.png")

        plt.close()

    """Print Check"""
    # print("")
    # print("Best number of clusters based on Silhouette Score:", best_grid['n_clusters'])
    # print("Best number of clusters based on the Elbow Method:", best_inertia_grid['n_clusters'])

    return best_grid['n_clusters']


def run_clustering_analysis(data, num_components_range, sample_names, dataset_name, path_to_output):
    """
    Purpose: This function performs the clustering analysis for the dataset by applying the selected dimension reduction
    technique and kmeans clustering. Multiple tables and plots are produced by this function.

    Input:
        1) PANDAS Dataframes: data
        2) Range: num_components_range
        3) Lists: sample_names
        3) Strs: dataset_name

    Output:
        1) PANDAS Dataframes: results
        2) Ints: cumulative_variance
        3) Lists: twoD_cluster_assignments
    """

    """
    Creating a random seed for the initialization of cluster centroids. This allows for more consistent and repeatable
    results. This basically just creates a constant to initialize the random number generator for k-means clustering.
    The value can be replaced by different integer values
    """
    random_seed = 150
    np.random.seed(random_seed)

    """Creating lists to store the optimum cluster, variance, pca weights and cluster assignments for each PCA test"""
    optimized_clusters = []
    variance = []
    pca_components_list = []
    cluster_data = []
    twoD_cluster_assignments = []

    """Creating a DataFrame to store the sample names for cluster assignments"""
    name_cluster_data = pd.DataFrame({
        'Sample Names': sample_names,
    })

    """This loops through all of the various PCA tests for the original dataset"""
    """YOU WILL NEED TO CHANGE THIS!"""
    with pd.ExcelWriter(path_to_output / 'Excel Files' / f'{dataset_name}' / f'{dataset_name}_Cluster_Assignments.xlsx',
                        engine='xlsxwriter') as writer:

        for num_components in num_components_range:

            if dataset_name == "LDA":
                result, cumulative_variance = lda_embeddings(data, sample_names, dataset_name, path_to_output,
                                                             num_components=num_components)
            elif dataset_name == "PCA":
                result, cumulative_variance = pca_embeddings(data, dataset_name, path_to_output,
                                                             num_components=num_components)

            elif dataset_name == "ISO":
                result, cumulative_variance = isomap_embeddings(data, path_to_output, num_components=num_components)

            elif dataset_name == "LLE":
                result, cumulative_variance = lle_embeddings(data, path_to_output, num_components=num_components)

            elif dataset_name == "TSNE":
                result, cumulative_variance = tsne_embeddings(data, path_to_output, num_components=num_components)

            elif dataset_name == "KPCA":
                result, cumulative_variance = kpca_embeddings(data, path_to_output, num_components=num_components,
                                                              kernel='rbf', gamma=None)

            """Checking for variance - we only want over 90% variance since the other PCA components do not provide
            a good model to base results off of"""
            # if cumulative_variance[-1] * 100 < 90:
            #   continue

            optimum_num_clusters = kmean_hyper_param_tuning(result, num_components, dataset_name, path_to_output)
            # print("Optimum num of clusters based on silhouette scores =", optimum_num_clusters)

            """Fitting KMeans"""
            # kmeans = KMeans(n_clusters=optimum_num_clusters, random_state=random_seed)  # for random seeding
            kmeans = KMeans(n_clusters=optimum_num_clusters)  # can hard code the number of clusters you want to use
            kmeans.fit(result)

            if num_components == 2:
                twoD_cluster_assignments.append(kmeans.labels_)

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

            """Print Check"""
            # print(f'\nCluster assignments for PCA {num_components}:')

            """Sorting the cluster data by cluster assignment"""
            individual_cluster_data['Cluster Assignments'] = kmeans.labels_
            individual_cluster_data_sorted = individual_cluster_data.sort_values(by='Cluster Assignments')

            sheet_name = f'Dim_{num_components}_Cluster_Assignments'
            individual_cluster_data_sorted.to_excel(writer, sheet_name=sheet_name, index=False)

    """Creating table to determine the best number of components to use"""
    table_data = {
        'Components': pca_components_list,
        'Optimized Clusters': optimized_clusters,
        'Variance Explained (%)': variance
    }

    table_data_df = pd.DataFrame(table_data)

    """Putting the dim red result into a dataframe to export the results"""
    results = pd.DataFrame(result)

    """Sorting the cluster data by cluster assignment"""
    name_cluster_data['Cluster Assignments'] = kmeans.labels_

    """Saving the clustering results in to excel files"""
    name_cluster_data.to_excel(path_to_output / 'Excel Files' / f'{dataset_name}' / f'Cluster_Assignments_{dataset_name}.xlsx')
    table_data_df.to_excel(path_to_output / 'Excel Files' / f'{dataset_name}' / f'Opt_Components_Table_{dataset_name}.xlsx')

    """Commented out because no longer using for analysis - would probably need to be updated to run properly"""
    # heatmap(cluster_data, sample_names, pca_components_list)

    return results, cumulative_variance, twoD_cluster_assignments


"""Analysis functions that do most of the computation and make the results of the ML techniques easier to 
visualize/digest. Essentially these functions create the main outputs that were utilized  in the analysis for the 
research journal and the thesis paper."""


def calculate_avg(data, ranges, avg_names):
    """
    Purpose:
    This function takes in a dataframe containing only the values for a specified polymer so that the values can be
    averaged to provide a "reference point" for distance calculations

    Input:
        1) PANDAS Dataframe: data
        2) List: ranges, avg_names

    Output:
        1) PANDAS Dataframe: result
        2) List: avg_names
    """

    """Initializing an array to store the averaged column values and standard deviations"""
    averaged_dataset = []
    one_std_dataset = []

    """Looping through ranges/blocks to average and find std for each individual polymer lot"""
    for start, end in ranges:
        if data.shape[1] == 1:  # Check if there is only one column in the DataFrame
            block = data.iloc[start:end + 1].values  # Using .values to get the values to average
        else:
            block = data.iloc[start:end + 1, :]

        """Getting the mean and std of each block (polymer lot block)"""
        avg_block = np.mean(block, axis=0)
        std_block = np.std(block, axis=0)

        """Putting into dataframes"""
        avg_block_df = pd.DataFrame(data=avg_block, columns=data.columns if data.shape[1] > 1 else [data.columns[0]])
        std_block_df = pd.DataFrame(data=std_block, columns=data.columns if data.shape[1] > 1 else [data.columns[0]])

        averaged_dataset.append(avg_block_df)
        one_std_dataset.append(std_block_df)

    """Concatenating the dataframes together"""
    result = pd.concat(
        [pd.concat(averaged_dataset, ignore_index=True),
         pd.concat(one_std_dataset, ignore_index=True)],
        axis=1
    )

    result.columns = [f'{col}_{stat}' for stat in ['Avg', 'Std'] for col in data.columns]

    """Print Check"""
    # print(result)

    return result, avg_names


def calculate_distances(data, datapoint, distance_metric='euclidean'):
    """
    Purpose: This function calculates distance between datapoints. There are multiple ways to do this but the most
    common are euclidean, cityblock, and cosine. This function can do any of these calculations but it is set for
    euclidean distance based on the input argument.

    Input: data, datapoint, distance_metric='euclidean'
        1) PANDAS Dataframe: data
        2) PANDAS Dataframe Element: datapoint
        3) Str: distance_metric

    Output:
        1) PANDAS Dataframe: distances
    """

    """Creating a list to store all the distance values"""
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


def pearson_correlation(result, eds_data, avg_names, cumulative_variance, dev_data, dataset_name, original_ranges,
                        path_to_output):
    """
    Purpose: This function calculates the correlation between the various PCA results and Ed's calculations (viscosity,
    crossover, frequency). We used it to determine the accuracy between eds calculations and the ML technique results.

    Input:
        1) PANDAS Dataframes: result, eds_data, dev_data
        2) Lists: avg_names, original_ranges
        3) Ints: cumulative_variance
        4) Strs: dataset_name

    Output:
        1) PANDAS Dataframes: correlations
        2) Union (tables): all_average_results, all_error_results
    """

    correlations = pd.DataFrame(index=[f'PCA_{i + 1}' for i in range(len(result.columns))], columns=eds_data.columns)
    all_average_results = pd.DataFrame()
    all_error_results = pd.DataFrame()

    """Creating a new set of ranges that can be updated as to keep the original ranges intact"""
    modified_ranges = original_ranges.copy()

    for col_result in result.columns:

        current_column_result = result[col_result]

        """Calculating the variance explained for the current PCA and calling the averaged_histogram"""
        if col_result == 0:  # setting the current variance to the first value in the cumulative_variance array
            current_variance = cumulative_variance[col_result]
        else:  # takes the cumulative_variance at the indexed value and subtracts the previous indices
            current_variance = cumulative_variance[col_result] - cumulative_variance[col_result - 1]

        """Print Check"""
        # print("\nThis is the current column at BEFORE of averaged_histogram function:\n", current_column_result)

        """Check for if there was no correlation"""
        averaged_values, error = averaged_histogram(result, current_column_result, col_result, original_ranges,
                                                    avg_names,
                                                    current_variance,
                                                    dev_data, dataset_name, path_to_output)

        all_average_results = pd.concat([all_average_results, averaged_values], ignore_index=True)
        all_error_results = pd.concat([all_error_results, error], ignore_index=True)

        """This loop is to loop through Ed's data to do the Pearson Correlation - checks the linear relationship 
        between our results - should be between -1 and 1 where -1 = negative linear, 0 = no correlation, and 
        1 = positive linear"""
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
                fig_pear_cor, ax = plt.subplots()
                ax.scatter(current_column, current_column_eds)
                ax.set_title(f'Scatter Plot: {dataset_name}_{result.columns.get_loc(col_result) + 1} vs {col_eds}')
                ax.set_xlabel(f'{dataset_name}_{result.columns.get_loc(col_result) + 1}')
                ax.set_ylabel(col_eds)
                ax.text(0.6, 0.9, f'Correlation: {correlation: .2f}', transform=plt.gca().transAxes, fontsize=10,
                        color='blue')

                """Making sure that there are no invalid characters in the filename"""
                scatter_filename = f'Scatter_{dataset_name}_{result.columns.get_loc(col_result) + 1}_vs_{col_eds}'
                sanitized_filename_scatter = sanitize_filename(scatter_filename)

                fig_pear_cor.savefig(path_to_output / 'Comparing Ed Data' / sanitized_filename_scatter)

                plt.close()

    return correlations, all_average_results, all_error_results


def averaged_histogram(result, current_column_result, col_result, input_range, avg_names, current_variance, dev_data,
                       dataset_name, path_to_output):
    """
    Purpose: Average the data points (PCA components) from the correlation graphs calculate the std and then plots them
    in a histogram with the std. x-axis is polymer name and y-axis is averaged pca value. (Used in the
    pearson_correlation function)

    Input:
        1) PANDAS Dataframes: result, dev_data
        2) PANDAS Series: current_column_result, col_result
        3) Lists: avg_names
        4) Ints: current_variance, input range
        5) Strs: dataset_name

    Output:
        1) PANDAS Dataframes: averaged_points, error
    """

    current_column_result = current_column_result.to_frame()  # current_column_pca from a pandas series to a DataFrame

    """Print Check"""
    # print("\nThis is the current column:\n", current_column_result)

    """Add check to see if more than one data point for each polymer to average (error bars)"""
    boolean = check_single_data_point_ranges(input_range)

    if boolean:  # If there is one data point per polymer - just tensile data error
        averaged_points = current_column_result
        error = pd.DataFrame(index=averaged_points.index, columns=averaged_points.columns)
        for index, row in averaged_points.iterrows():
            error.loc[index] = abs(dev_data.iloc[index, :].sum() * row)
        averaged_points = averaged_points.T
        error = error.T

        """Print Check"""
        # print('\nThese are the averaged values:', averaged_points)
        # print('\nThese are the error values:', error)

    else:  # If there is more than one data point per polymer
        if dev_data.empty:  # Just rheology data error
            averaged_pca, avg_names = calculate_avg(current_column_result, input_range, avg_names)
            averaged_points = averaged_pca.iloc[:, [0]].copy()
            error = averaged_pca.iloc[:, [1]].copy()
            averaged_points = averaged_points.T
            error = error.T

            """Print Check"""
            # print('\nThese are the averaged values:', averaged_points)
            # print('\nThese are the error values:', error)

        else:  # We stopped working with the combined dataset - this was never finished/corrected
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

    """Print Check"""
    # print(averaged_points)
    # print(error)

    """Creating the histogram"""
    fig_histogram_comp_ed, ax = plt.subplots(figsize=(3.75, 3.75))
    ax.bar(range(len(avg_names)), averaged_points.iloc[0],
           yerr=error.iloc[0], capsize=5, color='paleturquoise')
    # ax.set_title(f"Histogram: {dataset_name}{result.columns.get_loc(col_result) + 1}", fontsize=16)
    ax.set_ylabel(f"{dataset_name}{result.columns.get_loc(col_result) + 1}", fontsize=12)
    ax.set_xlabel("Polymer", fontsize=12)
    ax.set_xticks(range(len(avg_names)))
    ax.set_xticklabels(avg_names, rotation=90, fontsize=12)
    ax.set_yticklabels(ax.get_yticks(), fontsize=12)
    # ax.text(0.6, 1.1, f'Variance Explained: {current_variance: .2f}', transform=ax.transAxes, fontsize=12,
    # color='black') # Adding the variance explained to the histogram
    plt.tight_layout()

    histogram_filename = f'Histogram_{dataset_name}{result.columns.get_loc(col_result) + 1}'
    sanitized_filename_histogram = sanitize_filename(histogram_filename)

    fig_histogram_comp_ed.savefig(path_to_output / 'Comparing Ed Data' / sanitized_filename_histogram)

    plt.close()

    return averaged_points, error


def distance_radii(polymer_lots, result, ranges, dataset_name, num_components, path_to_output):
    """
    Purpose: This function calculates the distance and the radii (standard deviation) for each polymer lot to a lot of
    interest. (This is where similarity is quantified) This function does not return anything but produces a table with
    the findings.

    Input:
        1) PANDAS Dataframes: result
        2) Lists: polymer_lots, ranges
        3) Strs: dataset_name

    Output: N/A
    """

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

    """Calculating the radii for the polymer lot clusters - finding distances between centroid and each same polymer lot
     test"""
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

    """Print Check"""
    # print("This is the distances before scaling: \n")
    # print(result)

    """Scaling the Results to fall within 0 to 1"""
    scaler = MinMaxScaler()
    scaler.fit(result[['Distance from 30035']])
    result[['Distance from 30035']] = scaler.transform(result[['Distance from 30035']])

    """Getting the min and max values used by the scaler for 'Distance from 30035'"""
    distance_min = scaler.data_min_[0]
    distance_max = scaler.data_max_[0]

    """Print Check"""
    # print(distance_min)
    # print(distance_max)

    """Applying the same scaling to the 'Radii' column using the min and max from 'Distance from 30035'"""
    result['Radii'] = (result['Radii'] - distance_min) / (distance_max - distance_min)

    """Print Check"""
    # print("\nScaled Results:\n")
    # print(result)

    """Saving the DataFrame to an Excel file"""
    result.to_excel(path_to_output / 'Excel Files' / f'{dataset_name}' / f'Distance_Radii_Table_{dataset_name}.xlsx', sheet_name='sheet1',
                    index=False)

    """Print Check"""
    # print(result['Polymer'])
    # print(result['Distance from 30035'])
    # print(result['Radii'])

    fig_avg_distance_histo, ax = plt.subplots(figsize=(3.75, 3.75))
    ax.bar(result['Polymer'], result['Distance from 30035'],
           yerr=result['Radii'], capsize=5, color='paleturquoise')
    # ax.set_title(f"Histogram: {dataset_name}{result.columns.get_loc(col_result) + 1}", fontsize=16)
    ax.set_ylabel(f"Scaled Distance", fontsize=12)
    ax.set_xlabel("Polymer", fontsize=12)
    ax.set_xticks(result['Polymer'])
    ax.set_xticklabels(result['Polymer'], rotation=90, fontsize=12)
    ax.set_yticklabels(ax.get_yticks(), fontsize=12)
    ax.yaxis.set_major_formatter('{x:0<2.1f}')
    plt.tight_layout()
    # plt.show()

    histogram_filename = f'Distance_Histogram_{num_components}_Comp_{dataset_name}'
    sanitized_filename_histogram = sanitize_filename(histogram_filename)

    fig_avg_distance_histo.savefig(path_to_output / 'Comparing Ed Data' / sanitized_filename_histogram)


def plot_2D(sample_names, result, dataset_name, ranges, cluster_assignments, variance, path_to_output):
    """
    Purpose: This function creates 2D scatter plots for the PCA/LDA results and saves them to the computer.

    Input:
        1) PANDAS Dataframes: result
        2) Lists: sample_names, ranges, cluster_assignments, variance
        3) Strs: dataset_name

    Output: N/A
    """

    """Plotting the first two components"""
    # print(cluster_assignments)
    # print(dataset_name)
    # print(variance)

    captured_variance_1 = variance[0] * 100
    captured_variance_2 = (variance[1] - variance[0]) * 100

    colors = ['#BCBD22', '#1F77B4', '#FF7F0E', '#2CA02C', '#9467BD', '#8C564B', '#7F7F7F']
    cluster_colors = ['red', 'black', 'navy', 'magenta', 'cyan', 'gold', 'indigo']
    markers = ['*', 'x', 'D', '^', 'v', 'p', 'o']

    com_1 = result.loc[:, 0]
    com_2 = result.loc[:, 1]

    """Checking to see which dataset we are working with"""
    if dataset_name == "PCA":
        fig_2D_scatter, ax = plt.subplots(figsize=(5.5, 5.5))
        for index, (start, end) in enumerate(ranges):
            index_class = range(start, end + 1)
            ax.scatter(com_1.iloc[index_class], com_2.iloc[index_class], color=colors[index], marker=markers[index],
                       s=50,
                       label=f'Class: {sample_names[index]}')

        if cluster_assignments is not None:
            cluster_assignments = np.array(cluster_assignments).flatten()  # Flattening the array if it's nested
            for cluster in np.unique(cluster_assignments):
                cluster_points = (cluster_assignments == cluster)
                points = np.vstack((com_1[cluster_points], com_2[cluster_points])).T

                if len(points) < 3:
                    print(
                        f"Not enough points to construct a convex hull for cluster {cluster}. Drawing as points or "
                        f"lines.")
                    if len(points) == 1:
                        ax.scatter(points[:, 0], points[:, 1], color=cluster_colors[cluster % len(cluster_colors)],
                                   label=f'Cluster {cluster}')
                    elif len(points) == 2:
                        ax.plot(points[:, 0], points[:, 1], 'o-', color=cluster_colors[cluster % len(cluster_colors)],
                                label=f'Cluster {cluster}')
                else:
                    hull = ConvexHull(points)
                    for simplex in hull.simplices:
                        ax.plot(points[simplex, 0], points[simplex, 1],
                                color=cluster_colors[cluster % len(cluster_colors)])

                # ax.fill(points[hull.vertices, 0], points[hull.vertices, 1],
                #       color=cluster_colors[cluster % len(cluster_colors)], alpha=0.1)

        ax.legend(loc="best", shadow=False, scatterpoints=1)
        ax.set_title(f"2D {dataset_name} Results")
        ax.set_xlabel(f"{dataset_name}1 (Explained Variance: {captured_variance_1:.2f}%)", fontsize=12)
        ax.set_ylabel(f"{dataset_name}2 (Explained Variance: {captured_variance_2:.2f}%)", fontsize=12)

        """Changing the scales for the x- and y-axis so that all generated plots are on the same scale for comparison"""
        # ax.set_xlim(-15, 20)
        # ax.set_ylim(-4, 10)

        # ax.set_xlim(-100, 100)
        # ax.set_ylim(-60, 60)

    else:
        fig_2D_scatter, ax = plt.subplots(figsize=(5.5, 5.5))
        for index, (start, end) in enumerate(ranges):
            index_class = range(start, end + 1)
            ax.scatter(com_1.iloc[index_class], com_2.iloc[index_class], color=colors[index], marker=markers[index],
                       s=50,
                       label=f'Class: {sample_names[index]}')

        if cluster_assignments is not None:
            cluster_assignments = np.array(cluster_assignments).flatten()  # Flattening the array if it's nested
            for cluster in np.unique(cluster_assignments):
                cluster_points = (cluster_assignments == cluster)
                points = np.vstack((com_1[cluster_points], com_2[cluster_points])).T

                if len(points) < 3:
                    print(
                        f"Not enough points to construct a convex hull for cluster {cluster}. Drawing as points or lines.")
                    if len(points) == 1:
                        ax.scatter(points[:, 0], points[:, 1], color=cluster_colors[cluster % len(cluster_colors)],
                                   label=f'Cluster {cluster}')
                    elif len(points) == 2:
                        ax.plot(points[:, 0], points[:, 1], 'o-', color=cluster_colors[cluster % len(cluster_colors)],
                                label=f'Cluster {cluster}')
                else:
                    hull = ConvexHull(points)
                    for simplex in hull.simplices:
                        ax.plot(points[simplex, 0], points[simplex, 1],
                                color=cluster_colors[cluster % len(cluster_colors)])
                # ax.fill(points[hull.vertices, 0], points[hull.vertices, 1],
                #       color=cluster_colors[cluster % len(cluster_colors)], alpha=0.1)

        ax.legend(loc="best", shadow=False, scatterpoints=1)
        ax.set_title(f"2D {dataset_name} Results")
        ax.set_xlabel(f"{dataset_name}1 (Explained Discriminability: {captured_variance_1:.2f}%)", fontsize=12)
        ax.set_ylabel(f"{dataset_name}2 (Explained Discriminability: {captured_variance_2:.2f}%)", fontsize=12)

        """Changing the scales for the x- and y-axis so that all generated plots are on the same scale for comparison"""
        # ax.set_xlim(-300, 400)
        # ax.set_ylim(-80, 100)

        # ax.set_xlim(-20, 40)
        # ax.set_ylim(-15, 25)

    fig_2D_scatter.savefig(path_to_output / f'2D_{dataset_name}_Results.png')


"""Utility functions that are more general purpose."""


def shift_row(row):
    """
     Purpose: This function is used to shift a row within a dataset to the first valid index. (Used in
     preprocessing_tensile_data function)

     Input:
        1) row in PANDAS dataframe: row

     Output:
        1) row in PANDAS dataframe: row (shifted)
     """

    first_valid_index = row.first_valid_index()
    if first_valid_index is not None:
        num_shift = row.index.get_loc(first_valid_index)
        return row.shift(-num_shift)
    return row


def check_single_data_point_ranges(ranges):
    """
     Purpose: This function is used to check the range of the dataset to see if the data has already been averaged.

     Input:
        1) Lists: ranges

     Output:
        1) Boolean (true/false)
     """

    for start, end in ranges:
        if start != end:
            return False
    return True


def extract_label(sheet_name):
    """
     Purpose: This function is used to find the unique polymer lot names

     Input:
        1) Strs: sheet_name (A string containing the individual sheet_name within the Excel file)

     Output:
        1) Strs: label (A string with the unique polymer lot name. If it is not a unique name then the function returns
         None)
     """

    """Using regular expression to find the numbers in the sheet name"""
    match = re.search(r'\d+', sheet_name)
    if match:
        label = match.group()
        return label
    else:
        return None


def sanitize_filename(filename):
    """
    Purpose: This function is used to make sure that there are no invalid characters in filenames

    Input:
        1) Strs: filename

    Output:
        1) Strs: filename (without invalid characters)
    """

    invalid_chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename


"""I dont think I need these anymore"""


def save_plot(fig_or_plt, filename, dataset_name, folder_path=""):
    """
    Purpose: This function takes in a figure or a plot along with a filename and path to save each of the generated
    plots to a specified location on the computer (specified with folder path).

    Input:
        1) Strs: the plot object (either fig or plt), filename, the dataset name, and the folder_path

    Output: N/A
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
    Purpose: This function saves the clustering result tables to Excel files

    Input:
        1) Dictionary: table_data
        2) PANDAS Dataframe: name_cluster_data
        2) Strings: dataset_name and folder_path

    Output: N/A
    """

    """Creating specified filenames for different tables"""
    pca_components_filename = f'{dataset_name}_components_table.xlsx'
    pca_cluster_filename = f'{dataset_name}_cluster_table.xlsx'

    """Printing the clustered data to an Excel file"""
    name_cluster_data.to_excel(os.path.join(folder_path, pca_cluster_filename), sheet_name='sheet1', index=False)

    """Print Check"""
    # print("\nTable with all Component Cluster Assignments:")
    # print(name_cluster_data.to_string(index=False))

    """Creating DataFrame from table_data"""
    table_df = pd.DataFrame(table_data)

    """Print Check"""
    # print("\nTable with Components, Optimized Clusters, and Variance Explained:\n")
    # print(table_df.to_string(index=False))

    """Printing the tables output to an Excel file with a specific file path"""
    table_df.to_excel(os.path.join(folder_path, pca_components_filename), sheet_name='sheet1', index=False)


"""These additional functions probably won't be of any use to you. - Included them for additional reference but these 
avenues were abandoned in the research process. - Determined manifold learning techniques were not worthwhile for our
tensile datasets. The trends that were these techniques picked up on were not representative of the dataset but rather 
the noise in the datasets therefore we did not obtain accurate models/results."""


def kpca_embeddings(data, num_components, kernel='rbf', gamma=None):
    """
    Purpose: This function applies kernel principal component analysis (KPCA) to the dataset. (This was a short avenue
    taken - proved to be unhelpful and non-linear manifold learning techniques were not pursued rigorously)

    Input:
        1) PANDAS Dataframes: data (scaled)
        2) Ints: num_components
        3) kernel used: rbf - radial basis function
        4) kernel coefficient for rbf: gamma

    Output:
        1) kpca_results: This is a dataframe that contains the kernel principal component weights
        2) explained_variance_ratio: This is an integer containing the percentage (in decimal form) of the total trends
        captured within each of the kPCs
    """

    """Step 1: Applying KPCA"""
    kpca = KernelPCA(n_components=num_components, kernel=kernel, gamma=gamma)
    kpca_results = kpca.fit_transform(data)

    """Step 2: Computing the kernel matrix"""
    K = pairwise_kernels(data, metric=kernel, gamma=gamma)

    """Step 3: Centering the kernel matrix"""
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K_centered = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    """Step 4: Computing the eigenvalues"""
    eigvals, _ = np.linalg.eigh(K_centered)

    """Step 5: Sorting eigenvalues in descending order (they are returned in ascending order)"""
    eigvals = np.flip(eigvals)

    """Step 6: Computing the explained variance ratio"""
    explained_variance_ratio = eigvals / np.sum(eigvals)

    return kpca_results, explained_variance_ratio


def isomap_embeddings(data, num_components):
    """
    Purpose: This function applies isomap manifold learning to the dataset. (This was a short avenue
    taken - proved to be unhelpful and non-linear manifold learning techniques were not pursued rigorously)

    Input:
        1) PANDAS Dataframe: data (scaled)
        2) Ints: num_components

    Output:
        1) isomap_results: This is a dataframe that contains the transformed datapoints
        2) cumulative_variance_isomap: This is an integer containing the percentage (in decimal form) of the total
        trends captured
    """

    """Need to find the best number of neighbors to use"""
    n_neighbors_range = [2, 4, 6, 8, 10, 12]
    param_grid = {'n_neighbors': n_neighbors_range}

    """Setting n_neighbors to 7 since there are 7 types of tests (this may be wrong)"""
    isomap = Isomap(n_neighbors=20, n_components=num_components)

    # grid_search = GridSearchCV(isomap, param_grid, cv=5)
    # grid_search.fit(data)
    # best_isomap = grid_search.best_estimator_
    isomap_results = isomap.fit_transform(data)

    variances = np.var(isomap_results, axis=0)
    cumulative_variance_isomap = variances / np.sum(variances)

    # print("Best n_neighbors:", best_isomap.n_neighbors_)

    return isomap_results, cumulative_variance_isomap


def lle_embeddings(data, num_components):
    """
    Purpose: This function applies locally linear embedding manifold learning to the dataset. (This was a short avenue
    taken - proved to be unhelpful and non-linear manifold learning techniques were not pursued rigorously)

    Input:
        1) PANDAS Dataframe: data (scaled)
        2) Ints: num_components

    Output:
        1) lle_results: This is a dataframe that contains the transformed datapoints
        2) cumulative_variance_lle: This is an integer containing the percentage (in decimal form) of the total
        trends captured
    """

    lle = LocallyLinearEmbedding(n_components=num_components)
    lle_results = lle.fit_transform(data)
    variances = np.var(lle_results, axis=0)
    cumulative_variance_lle = variances / np.sum(variances)

    return lle_results, cumulative_variance_lle


def tsne_embeddings(data, num_components):
    """
    Purpose: This function applies t-distributed stochastic neighbor embedding manifold learning to the dataset.
    (This was a short avenue taken - proved to be unhelpful and non-linear manifold learning techniques were not pursued
     rigorously)

    Input:
        1) PANDAS Dataframe: data (scaled)
        2) Ints: num_components

    Output:
        1) tsne_results: This is a dataframe that contains the transformed datapoints
        2) cumulative_variance_tsne: This is an integer containing the percentage (in decimal form) of the total
        trends captured
    """

    tsne = TSNE(n_components=num_components)
    tsne_results = tsne.fit_transform(data)
    variances = np.var(tsne_results, axis=0)
    cumulative_variance_tsne = variances / np.sum(variances)

    return tsne_results, cumulative_variance_tsne


"""This function became no longer useful in later iterations. The heatmap function was used in the beginning of the 
project to visualize cluster assignments, however, as we developed our analysis further we looked into better ways to 
interpret results."""


def heatmap(cluster_data, sample_names, components_list):
    """
     Purpose: This function takes in the data from the clustering, the names of the polymers, and the number of PCs used
     and creates a heatmap for visualization of the clustering results using the imported module Plotly Express.
     (This probably won't be used - We replaced the visualization with the 2D plots - however Plotly Express is a great
     library for easy plotting!)

     Input:
        1) PANDAS Dataframes: cluster_data
        2) Lists: sample_names, components_list

     Output: N/A
     """

    """Transposing the cluster data"""
    cluster_data = pd.DataFrame(cluster_data).T

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


"""This function was never fully finished. I decided to keep it because it dives into the linear algebra utilized by 
the LDA algorithm. The goal was to trouble shoot some funky looking results that I got from a preliminary dataset. This 
 was abandoned because I figured out the problem before I was able to finish this function."""


def lda_hand_calculations(data, sample_names):
    """
    Purpose: This function attempts to perform LDA manually to calculate the eigenvectors for dimension reduction. This
    was performed to try and trouble shoot the lda_embeddings function at one point. (I would not worry about this
    function at all. I didn't even finish it because I fixed the previous problem without this.)

    Input:
        1) PANDAS Dataframes: data (scaled)
        2) Lists: sample_names

    Output: ???
    """

    """Rearranging results to match lba_embeddings results - (labels in descending order)"""
    # first_7_rows_names = sample_names[7:]
    # remaining_rows_names = sample_names[:7]
    # sample_names = first_7_rows_names + remaining_rows_names
    # print("\nThese are the NEW sample names:", sample_names)

    """Need to also move the first 7 rows in data to the last 7 to match with mean_vecs_reordered and sample_names"""
    # last_7_rows_data = data.iloc[7:]
    # remaining_rows_data = data.iloc[:7]
    # data = pd.concat([remaining_rows_data, last_7_rows_data])
    # print("Rearranged Data: \n", data)

    """Computing the d-dimensional mean vectors"""
    mean_vecs_list = []

    """Establishing new ranges since data was moved"""
    # ranges = [(0, 6), (7, 13), (14, 19), (20, 26), (27, 33), (34, 40), (41, 47)]
    ranges = [(0, 6), (7, 13), (14, 20), (21, 26), (27, 33), (34, 40), (41, 47)]

    for start, end in ranges:
        if data.shape[1] == 1:  # Check if there is only one column in the DataFrame
            block = data.iloc[start:end + 1].values  # Using .values to get the values to average
        else:
            block = data.iloc[start:end + 1, :]

        mean_vec = (np.mean(block, axis=0)).tolist()
        # print('This is the avg_block:', mean_vec)
        mean_vecs_list.append(mean_vec)

    """Storing in a dataframe"""
    mean_vecs = pd.DataFrame(mean_vecs_list)

    """Constructing the between-class (Sb) and within-class (Sw) scatter matrices"""

    number_features = data.shape[1]
    Sw = np.zeros((number_features, number_features))
    Sb = np.zeros((number_features, number_features))

    overall_mean = np.mean(data, axis=0)
    overall_mean = overall_mean.values
    max_classes = len(np.unique(sample_names))
    num_classes = range(1, max_classes + 1)

    """Within-class matrix"""
    for label, mv in zip(set(sample_names), mean_vecs.values):
        class_scatter = np.zeros((number_features, number_features))
        for row, name in zip(data.values, sample_names):
            if name == label:
                row = row.reshape(number_features, 1)
                mv = mv.reshape(number_features, 1)
                class_scatter += (row - mv).dot((row - mv).T)
        Sw += class_scatter

    """Between-class matrix this should be a 42 by 42 matrix"""
    for i, (mv, nc) in enumerate(zip(mean_vecs.values, num_classes)):
        # print("\nNum_classes: ", nc)
        overall_mean = overall_mean.reshape(-1, 1)  # make vector a column vector
        # print("\noverall_mean:", overall_mean)
        mv = mv.reshape(-1, 1)  # make vector a column vector
        # print("\nMean_vector: ", mv)
        Sb += nc * (mv - overall_mean).dot((mv - overall_mean).T)

    print('Within-class scatter matrix: \n', Sw)
    print('Within-class scatter matrix dimensions: %sx%s' % (Sw.shape[0], Sw.shape[1]))
    print('Between-class scatter matrix: \n', Sb)
    print('Between-class scatter matrix dimensions: %sx%s' % (Sb.shape[0], Sb.shape[1]))
    print('\nClass label distribution: %s' % np.bincount(sample_names)[1:])

    """Getting eigenvalues and eigenvectors"""
    eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    eigen_vals.shape, eigen_vecs.shape
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
    eigen_pairs = sorted(eigen_pairs, key=lambda x: x[0], reverse=True)
    print("\nHere are the eigenvalues in decreasing order: \n")
    for eigen_val in eigen_pairs:
        print(eigen_val[0])

    eigen_val_sum = sum(eigen_vals)
    print("\nExplained Variance:\n")
    for i, pair in enumerate(eigen_pairs):
        print('Eigenvector {}: {}'.format(i, (pair[0] / eigen_val_sum).real))

    """Calculating Y (matrix that contains the LDA components (new feature space))"""
    w_matrix = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
                          eigen_pairs[1][1][:, np.newaxis].real,
                          eigen_pairs[2][1][:, np.newaxis].real,
                          eigen_pairs[3][1][:, np.newaxis].real,
                          eigen_pairs[4][1][:, np.newaxis].real,
                          eigen_pairs[5][1][:, np.newaxis].real))

    eigenvectors_df = pd.DataFrame(w_matrix)
    print("\nHere are the eigenvectors hand-calc:\n ", w_matrix)

    eigenvectors_df.to_excel(r'C:\Users\tnt02\OneDrive\Documents\Masters Research\Running Data\Last Run\Excel '
                             r'Files\Eigenvectors LDA hand.xlsx', sheet_name='sheet1', index=False)

    # Sw.to_excel(r'C:\Users\tnt02\OneDrive\Desktop\Masters Research\Running Data\Last Run\Excel '
    #       r'Files\Within-Class Scatter Matrix hand calc.xlsx', sheet_name='sheet1', index=False)
