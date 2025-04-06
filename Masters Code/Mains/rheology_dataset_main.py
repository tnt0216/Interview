"""Importing packages"""
import pandas as pd
import warnings

import sklearn

from functions import load_embeddings_rheology, distance_radii
from functions import initial_data_graphs_rheology
from functions import calculate_avg
from functions import run_clustering_analysis
from functions import calculate_distances
from functions import pearson_correlation
from functions import plot_2D


def main():
    """Suppressing Warning Signs"""
    warnings.simplefilter(action='ignore', category=FutureWarning)  # This warning is because a variable changes
    warnings.simplefilter(action='ignore', category=RuntimeWarning)  # This warning is for generating 20+ pictures
    warnings.simplefilter(action='ignore', category=UserWarning)  # This warning occurs do to tight layout for some PCAs

    """Loading excel file"""
    # raw_data = "2055-2060 Series 70C Batch 2 Outliers.xlsx"
    raw_data = "2055-2060 Series 120C Batch 2 Outliers.xlsx"
    # raw_data = "2055-2060 Series 170C Batch 2 Outliers.xlsx"

    print("Loading Dataset")
    sample_names, concatenated_data, frequency_data, eds_data, num_datapoints = load_embeddings_rheology(raw_data)

    """Rearranging the rows so that 30035 data is first in both concatenated_data and eds_data"""
    last_7_rows_concat = concatenated_data.iloc[-6:]  # Grabbing 30035 rows
    remaining_rows_concat = concatenated_data.iloc[:-6]  # Grabbing the rest of the rows
    concatenated_data = pd.concat([last_7_rows_concat, remaining_rows_concat]).reset_index(drop=True)

    last_7_rows_names = sample_names[-6:]
    remaining_rows_names = sample_names[:-6]
    sample_names = last_7_rows_names + remaining_rows_names

    """These are only needed if you are comparing eds data"""
    # last_7_rows_eds = eds_data.iloc[-6:]  # Grabbing 30035 rows
    # remaining_rows_eds = eds_data.iloc[:-6]  # Grabbing the rest of the rows
    # eds_data = pd.concat([last_7_rows_eds, remaining_rows_eds]).reset_index(drop=True)

    print("After Shifting")
    print(concatenated_data)

    """Defining the ranges of rows to be averaged separately for the specific DataFrame - 
    THIS NEEDS TO BE ADJUSTED BASED ON WHAT DATASET YOU ARE RUNNING!"""
    # original_ranges = [(0, 5), (6, 11), (12, 17), (18, 24), (25, 30), (31, 36), (37, 42)]  # Outliers Dataset 70C
    original_ranges = [(0, 5), (6, 11), (12, 17), (18, 23), (24, 28), (29, 34), (35, 39)]  # Outliers Dataset 120C
    # original_ranges = [(0, 5), (6, 11), (12, 17), (18, 23), (24, 29), (30, 36), (37, 42)]  # Outliers Dataset 170C

    print("\nHere are the ranges for the polymer lots:\n", original_ranges)

    num_components = int(input('\n\nHow many components do you want to use?\n'))

    avg_names = ["30035", "2055", "2056", "2057", "2058", "2059", "2060"]

    """Naming the different datasets"""
    data0 = "Rheology"
    data1 = "PCA"
    data2 = "LDA"

    """Graphing storage and loss moduli and viscosity as a function of frequency"""
    # initial_data_graphs_rheology(concatenated_data, frequency_data, data0, num_datapoints)

    """HERE IS WHERE LDA STARTS"""

    """The number of components has to be num_classes - 1 (there are 7 classes since there are 7 polymer names therefore 
    the max number of components should be 6"""
    num_components_range_lda = range(1, 7)

    print("Running Clustering Analysis on Original Dataset (LDA):")
    lda_result, cumulative_variance_lda, twoD_cluster_assignments_LDA = run_clustering_analysis(concatenated_data,
                                                                                                num_components_range_lda,
                                                                                                sample_names, data2)

    print("This is the lda_result\n:", lda_result)
    print("\nThese are the variance ratios:\n", cumulative_variance_lda)

    """Plotting 2D results for LDA"""
    plot_2D(avg_names, lda_result, data2, original_ranges, twoD_cluster_assignments_LDA, cumulative_variance_lda)

    dev_data = pd.DataFrame()

    """Calculating pearson correlation values between lda results and Ed's calculations"""
    correlations_lda, averaged_values_lda, error_lda = pearson_correlation(lda_result, eds_data, avg_names,
                                                                           cumulative_variance_lda, dev_data, data2,
                                                                           original_ranges)

    error_lda.to_excel(r'C:\Users\tnt02\OneDrive\Documents\Masters Research\Running Data\Last Run\Excel '
                       r'Files\Error Calculations LDA.xlsx', sheet_name='sheet1', index=False)

    averaged_values_lda.to_excel(r'C:\Users\tnt02\OneDrive\Documents\Masters Research\Running Data\Last Run\Excel '
                                 r'Files\Average Calculations LDA.xlsx', sheet_name='sheet1', index=False)

    """We want to calculate the distances and standard deviations for the 2D plot polymer clusters"""
    lda_result = lda_result.iloc[:, :num_components]
    distance_radii(avg_names, lda_result, original_ranges, data2)

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
                                                                                                sample_names, data1)

    print("\nHere is PCA result: \n", pca_result)
    print("\nThese are the variance ratios:\n", cumulative_variance_pca)

    plot_2D(avg_names, pca_result, data1, original_ranges, twoD_cluster_assignments_PCA, cumulative_variance_pca)

    """Calculating pearson correlation values between PCA results and Ed's calculations"""
    correlations_pca, averaged_values_pca, error_pca = pearson_correlation(pca_result, eds_data, avg_names,
                                                                           cumulative_variance_pca,
                                                                           dev_data, data1, original_ranges)

    """Saving the DataFrame to an Excel file"""
    correlations_pca.to_excel(r'C:\Users\tnt02\OneDrive\Documents\Masters Research\Running Data\Last Run\Excel '
                              r'Files\Pearson Correlation Coefficients PCA.xlsx', sheet_name='sheet1', index=False)

    error_pca.to_excel(r'C:\Users\tnt02\OneDrive\Documents\Masters Research\Running Data\Last Run\Excel '
                       r'Files\Error Calculations PCA.xlsx', sheet_name='sheet1', index=False)

    averaged_values_pca.to_excel(r'C:\Users\tnt02\OneDrive\Documents\Masters Research\Running Data\Last Run\Excel '
                                 r'Files\Average Calculations PCA.xlsx', sheet_name='sheet1', index=False)

    """Distances for PCA"""
    pca_result = pca_result.iloc[:, :num_components]
    distance_radii(avg_names, pca_result, original_ranges, data1)


if __name__ == "__main__":
    main()
