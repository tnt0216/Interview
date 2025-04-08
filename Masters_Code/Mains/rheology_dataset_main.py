"""Importing packages"""
import pandas as pd
import warnings

from Masters_Code.functions import load_embeddings_rheology, distance_radii
from Masters_Code.functions import initial_data_graphs_rheology
from Masters_Code.functions import run_clustering_analysis
from Masters_Code.functions import pearson_correlation
from Masters_Code.functions import plot_2D
from Masters_Code.functions import making_output_folders
from Masters_Code.functions import path_to_input_rhe


def main():

    """Suppressing Warning Signs"""
    warnings.simplefilter(action='ignore', category=FutureWarning)  # This warning is because a variable changes
    warnings.simplefilter(action='ignore', category=RuntimeWarning)  # This warning is for generating 20+ pictures
    warnings.simplefilter(action='ignore', category=UserWarning)  # This warning occurs do to tight layout for some PCAs

    """Making output folders to store each type of output"""
    path_to_output = making_output_folders()

    """Loading excel file"""
    valid_file = False
    while not valid_file:

        file_chosen = str(input('\n\nWhich dataset would you like to run (70, 120, or 170)?\n'))

        if file_chosen == "70":
            valid_file = True
            dataset_chosen = "2055-2060 Series 70C Batch 2 Outliers.xlsx"
            raw_data = path_to_input_rhe / dataset_chosen
        elif file_chosen == "120":
            valid_file = True
            dataset_chosen = "2055-2060 Series 120C Batch 2 Outliers.xlsx"
            raw_data = path_to_input_rhe / dataset_chosen
        elif file_chosen == "170":
            valid_file = True
            dataset_chosen = "2055-2060 Series 170C Batch 2 Outliers.xlsx"
            raw_data = path_to_input_rhe / dataset_chosen
        else:
            print("Sorry that was not a valid dataset. Please try again.")

    print("You have chosen:", dataset_chosen)

    print("")
    print("Loading Dataset")
    sample_names, concatenated_data, frequency_data, eds_data, num_datapoints = load_embeddings_rheology(raw_data, dataset_chosen)

    """Rearranging the rows so that 30035 data is first in both concatenated_data and eds_data"""
    last_7_rows_concat = concatenated_data.iloc[-6:]  # Grabbing 30035 rows
    remaining_rows_concat = concatenated_data.iloc[:-6]  # Grabbing the rest of the rows
    concatenated_data = pd.concat([last_7_rows_concat, remaining_rows_concat]).reset_index(drop=True)

    last_7_rows_names = sample_names[-6:]
    remaining_rows_names = sample_names[:-6]
    sample_names = last_7_rows_names + remaining_rows_names

    """These are only needed if you are comparing eds data - added as the first sheet in excel file"""
    # last_7_rows_eds = eds_data.iloc[-6:]  # Grabbing 30035 rows
    # remaining_rows_eds = eds_data.iloc[:-6]  # Grabbing the rest of the rows
    # eds_data = pd.concat([last_7_rows_eds, remaining_rows_eds]).reset_index(drop=True)

    print("After Shifting")
    print(concatenated_data)
    print(dataset_chosen)

    """Defining the ranges of rows to be averaged separately for the specific DataFrame"""
    if dataset_chosen == '2055-2060 Series 70C Batch 2 Outliers.xlsx':
        original_ranges = [(0, 5), (6, 11), (12, 17), (18, 24), (25, 30), (31, 36), (37, 42)]  # Outliers Dataset 70C
    elif dataset_chosen == '2055-2060 Series 120C Batch 2 Outliers.xlsx':
        original_ranges = [(0, 5), (6, 11), (12, 17), (18, 23), (24, 28), (29, 34), (35, 39)]  # Outliers Dataset 120C
    else:
        original_ranges = [(0, 5), (6, 11), (12, 17), (18, 23), (24, 29), (30, 36), (37, 42)]  # Outliers Dataset 170C

    print("\nHere are the ranges for the polymer lots:\n", original_ranges)

    num_components = int(input('\n\nHow many components/dimensions do you want to use? (Choose between: 1-6)\n'))

    avg_names = ["30035", "2055", "2056", "2057", "2058", "2059", "2060"]

    """Naming the different datasets"""
    method2 = "PCA"
    method1 = "LDA"

    """Graphing storage and loss moduli and viscosity as a function of frequency"""
    initial_data_graphs_rheology(concatenated_data, frequency_data, num_datapoints, path_to_output)

    """HERE IS WHERE LDA STARTS"""

    """The number of components has to be num_classes - 1 (there are 7 classes since there are 7 polymer names therefore 
    the max number of components should be 6"""
    num_components_range_lda = range(1, 7)

    print("Running Clustering Analysis on Original Dataset (LDA):")
    lda_result, cumulative_variance_lda, twoD_cluster_assignments_LDA = run_clustering_analysis(concatenated_data,
                                                                                                num_components_range_lda,
                                                                                                sample_names, method1, path_to_output)

    print("This is the lda_result\n:", lda_result)
    print("\nThese are the variance ratios:\n", cumulative_variance_lda)

    """Plotting 2D results for LDA"""
    plot_2D(avg_names, lda_result, method1, original_ranges, twoD_cluster_assignments_LDA, cumulative_variance_lda, path_to_output)

    dev_data = pd.DataFrame()

    """Calculating pearson correlation values between lda results and Ed's calculations"""
    correlations_lda, averaged_values_lda, error_lda = pearson_correlation(lda_result, eds_data, avg_names,
                                                                           cumulative_variance_lda, dev_data, method1,
                                                                           original_ranges, path_to_output)

    correlations_lda.to_excel(path_to_output / 'Excel_Files' / 'LDA' / 'Pearson_Correlation_Coefficients_LDA.xlsx', sheet_name='sheet1', index=False)
    error_lda.to_excel(path_to_output / 'Excel_Files' / 'LDA' / 'Error_Calculations_LDA.xlsx', sheet_name='sheet1', index=False)
    averaged_values_lda.to_excel(path_to_output / 'Excel_Files' / 'LDA' / 'Average_Calculations_LDA.xlsx', sheet_name='sheet1', index=False)

    """We want to calculate the distances and standard deviations for the polymer clusters at the given num components"""
    lda_result = lda_result.iloc[:, :num_components]
    distance_radii(avg_names, lda_result, original_ranges, method1, num_components, path_to_output)

    """HERE IS WHERE PCA STARTS"""

    """Determining the maximum number of components based on the minimum of samples and dimensions and the range of PCA 
           components to iterate over for the original dataset (from 1 to the max number of components)"""
    max_components_pca = min(concatenated_data.shape[0], concatenated_data.shape[1])
    # num_components_range_pca = range(1, max_components_pca + 1)
    num_components_range_pca = range(1, 7)  # Only running 6 PCs

    """Need the clustering function to return the pca_result so that distance calculations can be made"""
    print("\nRunning Clustering Analysis on Original Dataset (PCA):")
    pca_result, cumulative_variance_pca, twoD_cluster_assignments_PCA = run_clustering_analysis(concatenated_data,
                                                                                                num_components_range_pca,
                                                                                                sample_names, method2, path_to_output)

    print("\nHere is PCA result: \n", pca_result)
    print("\nThese are the variance ratios:\n", cumulative_variance_pca)

    plot_2D(avg_names, pca_result, method2, original_ranges, twoD_cluster_assignments_PCA, cumulative_variance_pca, path_to_output)

    """Calculating pearson correlation values between PCA results and Ed's calculations"""
    correlations_pca, averaged_values_pca, error_pca = pearson_correlation(pca_result, eds_data, avg_names,
                                                                           cumulative_variance_pca,
                                                                           dev_data, method2, original_ranges, path_to_output)

    """Saving the DataFrame to an Excel file"""
    correlations_pca.to_excel(path_to_output / 'Excel_Files' / 'PCA' / 'Pearson_Correlation_Coefficients_PCA.xlsx', sheet_name='sheet1', index=False)
    error_pca.to_excel(path_to_output / 'Excel_Files' / 'PCA' / 'Error_Calculations_PCA.xlsx', sheet_name='sheet1', index=False)
    averaged_values_pca.to_excel(path_to_output / 'Excel_Files' / 'PCA' / 'Average_Calculations_PCA.xlsx', sheet_name='sheet1', index=False)

    """We want to calculate the distances and standard deviations for the polymer clusters at the given num components"""
    pca_result = pca_result.iloc[:, :num_components]
    distance_radii(avg_names, pca_result, original_ranges, method2, num_components, path_to_output)


if __name__ == "__main__":
    main()
