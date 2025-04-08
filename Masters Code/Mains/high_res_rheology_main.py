"""Importing packages"""
import pandas as pd
import warnings

from functions import load_embeddings_rheology
from functions import initial_data_graphs_high_res_rheology
from functions import run_clustering_analysis
from functions import pearson_correlation
from functions import plot_2D
from functions import distance_radii
from functions import making_output_folders
from functions import path_to_input_rhe


def main():

    """Suppressing Warning Signs"""
    warnings.simplefilter(action='ignore', category=FutureWarning)  # This warning is because a variable changes
    warnings.simplefilter(action='ignore', category=RuntimeWarning)  # This warning is for generating 20+ pictures
    warnings.simplefilter(action='ignore', category=UserWarning)  # This warning occurs do to tight layout for some PCAs

    """Making output folders to store each type of output"""
    path_to_output = making_output_folders()

    dataset_chosen = "High Precision 2055-2060 Series 170C.xlsx"

    print("Loading Dataset")
    raw_data = path_to_input_rhe / 'High Precision 2055-2060 Series 170C.xlsx'
    sample_names, concatenated_data, frequency_data, eds_data, num_datapoints = load_embeddings_rheology(raw_data, dataset_chosen)

    print(sample_names)

    """Defining the ranges of rows to be averaged separately for the specific DataFrame"""
    original_ranges = [(0, 6), (7, 10), (11, 17), (18, 24), (25, 31), (32, 38), (39, 45)]
    avg_names = ["30035", "2055", "2056", "2057", "2058", "2059", "2060"]

    num_components = int(input('\n\nHow many components do you want to use?\n'))

    """Naming the different dataset outputs"""
    data0 = "High Precision Rheology"
    method1 = "LDA"
    method2 = "PCA"

    """Graphing storage and loss moduli and viscosity as a function of frequency"""
    initial_data_graphs_high_res_rheology(concatenated_data, frequency_data, data0, num_datapoints, original_ranges,
                                          path_to_output)

    """HERE IS WHERE LDA STARTS"""

    """The number of components has to be num_classes - 1 (there are 4 classes since there are 4 polymer names therefore 
    the max number of components should be 3"""
    num_components_range_lda = range(1, 5)

    print("Running Clustering Analysis on Original Dataset (LDA):")
    lda_result, cumulative_variance_lda, twoD_cluster_assignments_LDA = run_clustering_analysis(concatenated_data,
                                                                                            num_components_range_lda,
                                                                                            sample_names, method1,
                                                                                            path_to_output)

    print("This is the lda_result\n:", lda_result)
    print("These are the variance ratios:\n", cumulative_variance_lda)

    """Plotting 2D results for LDA"""
    plot_2D(avg_names, lda_result, method1, original_ranges, twoD_cluster_assignments_LDA, cumulative_variance_lda,
            path_to_output)

    dev_data = pd.DataFrame()

    """Calculating pearson correlation values between lda results and Ed's calculations"""
    correlations_lda, averaged_values_lda, error_lda = pearson_correlation(lda_result, eds_data, avg_names,
                                                                           cumulative_variance_lda, dev_data, method1,
                                                                           original_ranges, path_to_output)

    correlations_lda.to_excel(path_to_output / 'Excel Files' / 'Pearson Correlation Coefficients LDA.xlsx',
                              sheet_name='sheet1', index=False)
    error_lda.to_excel(path_to_output / 'Excel Files' / 'Error Calculations LDA.xlsx', sheet_name='sheet1', index=False)
    averaged_values_lda.to_excel(path_to_output / 'Excel Files' / 'Average Calculations LDA.xlsx', sheet_name='sheet1',
                                 index=False)

    """We want to calculate the distances and standard deviations for the polymer cluster at the given num components"""
    lda_result = lda_result.iloc[:, :num_components]
    distance_radii(avg_names, lda_result, original_ranges, method1, num_components, path_to_output)

    """HERE IS WHERE PCA STARTS"""

    """Determining the maximum number of components based on the minimum of samples and dimensions and the range of PCA 
           components to iterate over for the original dataset (from 1 to the max number of components"""
    # max_components_pca = min(concatenated_data.shape[0], concatenated_data.shape[1])
    # num_components_range_pca = range(1, max_components_pca + 1)
    num_components_range_pca = range(1, 5)  # We only want to run to 4 dim since the 5 dim adds too much noise

    """Need the clustering function to return the pca_result so that distance calculations can be made"""
    print("\nRunning Clustering Analysis on Original Dataset (PCA):")
    pca_result, cumulative_variance_pca, twoD_cluster_assignments_PCA = run_clustering_analysis(concatenated_data, num_components_range_pca,
                                                                  sample_names, method2, path_to_output)

    print("\nHere is PCA result: \n", pca_result)
    print("\nHere are the ranges passed in to 2D plot:\n", original_ranges)

    plot_2D(avg_names, pca_result, method2, original_ranges, twoD_cluster_assignments_PCA, cumulative_variance_lda,
            path_to_output)

    """Calculating pearson correlation values between PCA results and Ed's calculations"""
    correlations_pca, averaged_values_pca, error_pca = pearson_correlation(pca_result, eds_data, avg_names,
                                                                           cumulative_variance_pca,
                                                                           dev_data, method2, original_ranges, path_to_output)

    correlations_pca.to_excel(path_to_output / 'Excel Files' / 'Pearson Correlation Coefficients PCA.xlsx',
                              sheet_name='sheet1', index=False)
    error_pca.to_excel(path_to_output / 'Excel Files' / 'Error Calculations PCA.xlsx', sheet_name='sheet1', index=False)
    averaged_values_pca.to_excel(path_to_output / 'Excel Files' / 'Average Calculations PCA.xlsx', sheet_name='sheet1',
                                 index=False)

    """Distances for PCA"""
    pca_result = pca_result.iloc[:, :num_components]
    distance_radii(avg_names, pca_result, original_ranges, method2, num_components, path_to_output)


if __name__ == "__main__":
    main()
