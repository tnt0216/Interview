"""Importing packages"""
import pandas as pd
import warnings

from sklearn.preprocessing import StandardScaler
from functions import load_embeddings_tensile
from functions import run_clustering_analysis
from functions import plot_2D
from functions import pearson_correlation
from functions import distance_radii
from functions import stress_strain_plots
from functions import preprocessing_tensile_data
from functions import interpolation


def main():

    """Suppressing Warning Signs"""
    warnings.simplefilter(action='ignore', category=FutureWarning)  # This warning is because a variable changes
    warnings.simplefilter(action='ignore', category=RuntimeWarning)  # This warning is for generating 20+ pictures
    warnings.simplefilter(action='ignore', category=UserWarning)  # This warning occurs do to tight layout for some PCAs

    """Loading excel file"""
    # raw_data = "Tensile Data 2055-2060 24C.xlsx"
    raw_data = "Tensile Data 2055-2060 40C.xlsx"
    # raw_data = "Tensile Data 2055-2060 50C.xlsx"
    # raw_data = "Tensile Data 2055-2060 70C.xlsx"

    """Creating ranges for the different files"""
    # original_ranges = [(0, 3), (4, 10), (11, 17), (18, 24), (25, 30), (31, 36), (37, 43)]  # 24C
    original_ranges = [(0, 5), (6, 11), (12, 17), (18, 23), (24, 29), (30, 36), (37, 43)]  # 40C
    # original_ranges = [(0, 5), (6, 12), (13, 19), (20, 25), (26, 32), (33, 38), (39, 45)]  # 50C
    # original_ranges = [(0, 5), (6, 12), (13, 19), (20, 24), (25, 29), (30, 35)]  # 70C

    """Creating naming conventions for preprocessing"""
    data0 = "Raw"
    data1 = "Shifted"
    data2 = "Interpolated"
    avg_names = ["30035", "2055", "2056", "2057", "2058", "2059", "2060"]
    # avg_names = ["30035", "2055", "2056", "2057", "2058", "2059"]

    print("Loading Dataset")
    sample_names, stress_data, strain_data, force_data, num_datapoints, eds_data = load_embeddings_tensile(raw_data)
    shifted_stress_data, shifted_strain_data = preprocessing_tensile_data(force_data, stress_data, strain_data)

    interpolated_strain, interpolated_stress = interpolation(shifted_stress_data, shifted_strain_data)

    """Creating Stress Strain Plots for raw, shifted, and interpolated data"""
    stress_strain_plots(sample_names, stress_data, strain_data, data0, original_ranges)
    stress_strain_plots(sample_names, shifted_stress_data, shifted_strain_data, data1, original_ranges)
    stress_strain_plots(sample_names, interpolated_stress, interpolated_strain, data2, original_ranges)

    # Applying StandardScaler
    scaler = StandardScaler()
    stress_data_scaled = scaler.fit_transform(interpolated_stress)
    stress_data_scaled = pd.DataFrame(stress_data_scaled, columns=interpolated_stress.columns)

    method1 = "LDA"
    method2 = "PCA"

    method3 = "ISO"
    method4 = "LLE"
    method5 = "TSNE"

    method6 = "KPCA"

    """HERE IS MANIFOLD LEARNING METHODS
    # ISO METHOD
    num_components_range_iso = range(1, 5)
    print("Running Clustering Analysis on Original Dataset (ISO):")
    isomap_result, cumulative_variance_iso = run_clustering_analysis(stress_data_scaled, num_components_range_iso,
                                                                  sample_names, method3)
    print("This is the iso_result \n:", isomap_result)
    print("These are the variance ratios:\n", cumulative_variance_iso)

    # Plotting 2D results for LDA
    plot_2D(avg_names, isomap_result, method3, original_ranges)
    """

    """LLE METHOD
    num_components_range_lle = range(1, 5)
    print("Running Clustering Analysis on Original Dataset (ISO):")
    lle_result, cumulative_variance_lle = run_clustering_analysis(stress_data_scaled, num_components_range_lle,
                                                                     sample_names, method4)
    print("This is the lle_result \n:", lle_result)
    print("These are the variance ratios:\n", cumulative_variance_lle)

    # Plotting 2D results for LDA
    plot_2D(avg_names, lle_result, method4, original_ranges)
    """

    """T-SNE METHOD
    # Cannot set to over 4 dimensions because of the Barnes-Hut approximation
    num_components_range_tsne = range(1, 4)
    print("Running Clustering Analysis on Original Dataset (ISO):")
    tsne_result, cumulative_variance_tsne = run_clustering_analysis(stress_data_scaled, num_components_range_tsne,
                                                                  sample_names, method5)
    print("This is the tsne_result \n:", tsne_result)
    print("These are the variance ratios:\n", cumulative_variance_tsne)

    # Plotting 2D results for LDA
    plot_2D(avg_names, tsne_result, method5, original_ranges)
    """

    """HERE IS WHERE LDA STARTS"""

    num_components_range_lda = range(1, 5)  # We only want to run to 4 dim since the 5 dim adds too much noise
    print("Running Clustering Analysis on Original Dataset (LDA):")
    lda_result, cumulative_variance_lda, twoD_cluster_assignments_lda = run_clustering_analysis(stress_data_scaled, num_components_range_lda,
                                                                  sample_names, method1)

    print("This is the lda_result:\n", lda_result)
    print("These are the variance ratios:\n", cumulative_variance_lda)

    """Plotting 2D results for LDA"""
    # plot_2D(avg_names, lda_result, method1, original_ranges)
    plot_2D(avg_names, lda_result, method1, original_ranges, twoD_cluster_assignments_lda, cumulative_variance_lda)

    dev_data = pd.DataFrame()

    """Calculating pearson correlation values between PCA results and Ed's calculations"""
    correlations_lda, averaged_values_lda, error_lda = pearson_correlation(lda_result, eds_data, avg_names,
                                                                           cumulative_variance_lda,
                                                                           dev_data, method1, original_ranges)

    print("\nCorrelations: ")
    print(correlations_lda)

    """Saving the DataFrame to an Excel file"""
    correlations_lda.to_excel(r'C:\Users\tnt02\OneDrive\Documents\Masters Research\Running Data\Last Run\Excel '
                              r'Files\Pearson Correlation Coefficients PCA.xlsx', sheet_name='sheet1', index=False)

    error_lda.to_excel(r'C:\Users\tnt02\OneDrive\Documents\Masters Research\Running Data\Last Run\Excel '
                       r'Files\Error Calculations PCA.xlsx', sheet_name='sheet1', index=False)

    averaged_values_lda.to_excel(r'C:\Users\tnt02\OneDrive\Documents\Masters Research\Running Data\Last Run\Excel '
                                 r'Files\Average Calculations PCA.xlsx', sheet_name='sheet1', index=False)

    """Distances for LDA"""
    num_components = int(input('\n\nHow many components do you want to use?\n'))
    lda_result = lda_result.iloc[:, :num_components]
    distance_radii(avg_names, lda_result, original_ranges, method1)

    """HERE IS WHERE PCA STARTS"""

    # max_components_pca = min(stress_data_scaled.shape[0], stress_data_scaled.shape[1])
    # num_components_range_pca = range(1, max_components_pca + 1
    num_components_range_pca = range(1, 5)  # We only want to run to 4 dim since the 5 dim adds too much noise

    """Need the clustering function to return the pca_result so that distance calculations can be made"""
    print("\nRunning Clustering Analysis on Original Dataset (PCA):")
    pca_result, cumulative_variance_pca, twoD_cluster_assignments_pca = run_clustering_analysis(stress_data_scaled, num_components_range_pca,
                                                                  sample_names, method2)

    print("\nHere is PCA result: \n", pca_result)
    print("\nHere are the ranges passed in to 2D plot:\n", original_ranges)

    plot_2D(avg_names, pca_result, method2, original_ranges, twoD_cluster_assignments_pca, cumulative_variance_pca)

    """Calculating pearson correlation values between PCA results and Ed's calculations"""
    correlations_pca, averaged_values_pca, error_pca = pearson_correlation(pca_result, eds_data, avg_names,
                                                                           cumulative_variance_pca,
                                                                           dev_data, method2, original_ranges)

    print("\nCorrelations: ")
    print(correlations_pca)

    """Saving the DataFrame to an Excel file"""
    correlations_pca.to_excel(r'C:\Users\tnt02\OneDrive\Documents\Masters Research\Running Data\Last Run\Excel '
                              r'Files\Pearson Correlation Coefficients PCA.xlsx', sheet_name='sheet1', index=False)

    error_pca.to_excel(r'C:\Users\tnt02\OneDrive\Documents\Masters Research\Running Data\Last Run\Excel '
                       r'Files\Error Calculations PCA.xlsx', sheet_name='sheet1', index=False)

    averaged_values_pca.to_excel(r'C:\Users\tnt02\OneDrive\Documents\Masters Research\Running Data\Last Run\Excel '
                                 r'Files\Average Calculations PCA.xlsx', sheet_name='sheet1', index=False)

    """Distances for PCA"""
    pca_result = pca_result.iloc[:, :num_components]
    distance_radii(avg_names, pca_result, original_ranges, method2)

    """
    num_components_range_kpca = range(1, 5)  # We only want to run to 4 dim since the 5 dim adds too much noise
    print("Running Clustering Analysis on Original Dataset (KPCA):")

    kpca_result, cumulative_variance_kpca = run_clustering_analysis(stress_data_scaled, num_components_range_kpca,
                                                                  sample_names, method6)

    print("This is the kpca_result:\n", kpca_result)
    print("These are the variance ratios:\n", cumulative_variance_kpca)

    plot_2D(avg_names, kpca_result, method6, original_ranges)
    """


if __name__ == "__main__":
    main()
