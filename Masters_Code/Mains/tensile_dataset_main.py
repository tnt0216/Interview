"""Importing packages"""
import pandas as pd
import warnings

from sklearn.preprocessing import StandardScaler
from Masters_Code.functions import load_embeddings_tensile
from Masters_Code.functions import run_clustering_analysis
from Masters_Code.functions import plot_2D
from Masters_Code.functions import pearson_correlation
from Masters_Code.functions import distance_radii
from Masters_Code.functions import stress_strain_plots
from Masters_Code.functions import preprocessing_tensile_data
from Masters_Code.functions import interpolation
from Masters_Code.functions import making_output_folders
from Masters_Code.functions import path_to_input_ten


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

        file_chosen = str(input('\n\nWhich dataset would you like to run (24, 40, 50, or 70)?'))

        if file_chosen == "24":
            valid_file = True
            dataset_chosen = "Tensile Data 2055-2060 24C.xlsx"
            raw_data = path_to_input_ten / dataset_chosen
        elif file_chosen == "40":
            valid_file = True
            dataset_chosen = "Tensile Data 2055-2060 40C.xlsx"
            raw_data = path_to_input_ten / dataset_chosen
        elif file_chosen == "50":
            valid_file = True
            dataset_chosen = "Tensile Data 2055-2060 50C.xlsx"
            raw_data = path_to_input_ten / dataset_chosen
        elif file_chosen == "70":
            valid_file = True
            dataset_chosen = "Tensile Data 2055-2060 70C.xlsx"
            raw_data = path_to_input_ten / dataset_chosen
        else:
            print("Sorry that was not a valid dataset. Please try again.")

    print("You have chosen:", dataset_chosen)

    avg_names = ["30035", "2055", "2056", "2057", "2058", "2059", "2060"]

    """Creating ranges for the different files"""
    if dataset_chosen == "Tensile Data 2055-2060 24C.xlsx":
        original_ranges = [(0, 3), (4, 10), (11, 17), (18, 24), (25, 30), (31, 36), (37, 43)]  # 24C
    elif dataset_chosen == "Tensile Data 2055-2060 40C.xlsx":
        original_ranges = [(0, 5), (6, 11), (12, 17), (18, 23), (24, 29), (30, 36), (37, 43)]  # 40C
    elif dataset_chosen == "Tensile Data 2055-2060 50C.xlsx":
        original_ranges = [(0, 5), (6, 12), (13, 19), (20, 25), (26, 32), (33, 38), (39, 45)]  # 50C
    else:
        original_ranges = [(0, 5), (6, 12), (13, 19), (20, 24), (25, 29), (30, 35)]  # 70C
        avg_names = ["30035", "2055", "2056", "2057", "2058", "2059"]  # This dataset doesn't contain data for lot 2060

    """Creating naming conventions for preprocessing"""
    data0 = "Raw"
    data1 = "Shifted"
    data2 = "Interpolated"

    print("")
    print("Loading Dataset")
    sample_names, stress_data, strain_data, force_data, num_datapoints, eds_data = load_embeddings_tensile(raw_data)
    shifted_stress_data, shifted_strain_data = preprocessing_tensile_data(force_data, stress_data, strain_data)
    interpolated_strain, interpolated_stress = interpolation(shifted_stress_data, shifted_strain_data)

    """Creating Stress Strain Plots for raw, shifted, and interpolated data"""
    stress_strain_plots(sample_names, stress_data, strain_data, data0, original_ranges, path_to_output)
    stress_strain_plots(sample_names, shifted_stress_data, shifted_strain_data, data1, original_ranges, path_to_output)
    stress_strain_plots(sample_names, interpolated_stress, interpolated_strain, data2, original_ranges, path_to_output)

    """Applying StandardScaler"""
    scaler = StandardScaler()
    stress_data_scaled = scaler.fit_transform(interpolated_stress)
    stress_data_scaled = pd.DataFrame(stress_data_scaled, columns=interpolated_stress.columns)

    method1 = "LDA"
    method2 = "PCA"

    num_components = int(input('\n\nHow many components do you want to use?\n'))

    """HERE IS WHERE LDA STARTS"""

    num_components_range_lda = range(1, 5)  # We only want to run to 4 dim since the 5 dim adds too much noise
    print("Running Clustering Analysis on Original Dataset (LDA):")
    lda_result, cumulative_variance_lda, twoD_cluster_assignments_lda = run_clustering_analysis(stress_data_scaled, num_components_range_lda,
                                                                  sample_names, method1, path_to_output)

    print("This is the lda_result:\n", lda_result)
    print("These are the variance ratios:\n", cumulative_variance_lda)

    """Plotting 2D results for LDA"""
    plot_2D(avg_names, lda_result, method1, original_ranges, twoD_cluster_assignments_lda, cumulative_variance_lda, path_to_output)

    dev_data = pd.DataFrame()

    """Calculating pearson correlation values between PCA results and Ed's calculations"""
    correlations_lda, averaged_values_lda, error_lda = pearson_correlation(lda_result, eds_data, avg_names,
                                                                           cumulative_variance_lda,
                                                                           dev_data, method1, original_ranges, path_to_output)

    """Saving results to Excel files"""
    correlations_lda.to_excel(path_to_output / 'Excel_Files' / 'LDA' / 'Pearson_Correlation_Coefficients_LDA.xlsx',
                              sheet_name='sheet1', index=False)
    error_lda.to_excel(path_to_output / 'Excel_Files' / 'LDA' / 'Error_Calculations_LDA.xlsx', sheet_name='sheet1',
                       index=False)
    averaged_values_lda.to_excel(path_to_output / 'Excel_Files' / 'LDA' / 'Average_Calculations_LDA.xlsx',
                                 sheet_name='sheet1', index=False)

    """Distances for LDA"""
    lda_result = lda_result.iloc[:, :num_components]
    distance_radii(avg_names, lda_result, original_ranges, method1, num_components, path_to_output)

    """HERE IS WHERE PCA STARTS"""

    # max_components_pca = min(stress_data_scaled.shape[0], stress_data_scaled.shape[1])
    # num_components_range_pca = range(1, max_components_pca + 1
    num_components_range_pca = range(1, 5)  # We only want to run to 4 dim since the 5 dim adds too much noise

    """Need the clustering function to return the pca_result so that distance calculations can be made"""
    print("\nRunning Clustering Analysis on Original Dataset (PCA):")
    pca_result, cumulative_variance_pca, twoD_cluster_assignments_pca = run_clustering_analysis(stress_data_scaled, num_components_range_pca,
                                                                  sample_names, method2, path_to_output)

    print("\nHere is PCA result: \n", pca_result)
    print("\nHere are the ranges passed in to 2D plot:\n", original_ranges)

    plot_2D(avg_names, pca_result, method2, original_ranges, twoD_cluster_assignments_pca, cumulative_variance_pca, path_to_output)

    """Calculating pearson correlation values between PCA results and Ed's calculations"""
    correlations_pca, averaged_values_pca, error_pca = pearson_correlation(pca_result, eds_data, avg_names,
                                                                           cumulative_variance_pca,
                                                                           dev_data, method2, original_ranges, path_to_output)

    """Saving the results to Excel files"""
    correlations_pca.to_excel(path_to_output / 'Excel_Files' / 'PCA' / 'Pearson_Correlation_Coefficients_LDA.xlsx',
                              sheet_name='sheet1', index=False)
    error_pca.to_excel(path_to_output / 'Excel_Files' / 'PCA' / 'Error_Calculations_LDA.xlsx', sheet_name='sheet1',
                       index=False)
    averaged_values_pca.to_excel(path_to_output / 'Excel_Files' / 'PCA' / 'Average_Calculations_LDA.xlsx',
                                 sheet_name='sheet1', index=False)

    """Distances for PCA"""
    pca_result = pca_result.iloc[:, :num_components]
    distance_radii(avg_names, pca_result, original_ranges, method2, num_components, path_to_output)


if __name__ == "__main__":
    main()
