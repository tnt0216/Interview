Master's Research Project
=========================

PROJECT DESCRIPTION:
--------------------
The purpose of this project is to assess the rheological and tensile similarity between various PVDF-CTFE legacy lots with varying synthesis and processing techniques. Dimension reduction (PCA / LDA) was used to reduce datasets containing 100+ dimensions to 2 dimensions to qualitatively assess similarity. Distance calculations were also made to quantitatively assess similarity. Clustering (k-means) was also applied to group similar polymer lots together to further verify similarities and assess outliers. Only the final versions of code are contained within this repo for viewing.

VERSION REQUIREMENTS:
--------------------
These scripts were written using Python 3.9.13. I have included the requirements.txt for the environment set up with pip.

NAVIGATING FOLDERS:
-------------------
In the "Masters Code" folder, there are subfolders titled "Additional Graphs", "Excel Files", and "Mains". 

Additional Graphs: Contains a few short scripts for additional graphs we wanted to use in our papers.
  
Excel Files: Contains all of the final data used for our scientific journal and my thesis. Subfolders are separating the rheological and tensile datasets.

About the Datasets:
- The rheology datasets were the primary datasets used in our analysis and were the focus of the research journal and thesis. There were three rheology datasets for temperatures of 70°C, 120°C, and 170°C used by "rheology_dataset_main". Each of these datasets had undergone data preprocessing by applying Chavunet's Criteria to identify outliers in the dataset. These outliers were taken completely out before running the data through these scripts. We compared our results to an experimental approach of assessing rheological similarity through the crossover point and zero-shear viscosity. These outliers were not included and handled in this code because we wanted to have identical datasets to the experimental approach for direct comparisons of performance.
- We investigated the impact of high-frequency data on our results to determine if it was worthwhile to perform larger frequency sweeps. We determined that it was not worthwhile to pursue, however, the code and high-res dataset have been included and are referred to as "high_res_rheology_main" and "High Precision 2055-2060 Series 170C.xlsx".
- Four temperature tensile datasets were collected for analysis, which were 24°C, 40°C, 50°C, and 70°C, respectively. These datasets had to be adjusted in preprocessing so that they did not contain negative values, each test contained the same number of sampled datapoints/dimensions, and each of the datapoints being compared to each other corresponded to the same strain value, the first dimension/first value in each test was set to zero and following datapoints were shifted accordingly. 

Mains: This folder contains the main functions/main scripts.

Functions: This file contains all the various functions that each of the main scripts executes. 

HOW TO USE:
---------------
1) Download appropriate files: the function file (used by each main), the main you would like to run (rheology_dataset_main, high_res_rheology_main, or tensile_dataset_main), and the dataset/Excel file.

Notes:
- When running "rheology_dataset_main": The default dataset has been set to "2055-2060 Series 120C Batch 2 Outliers.xlsx".
- When running "high_res_rheology_main": The corresponding dataset is titled "High Precision 2055-2060 Series 170C.xlsx".
- When running "tensile_dataset_main": The default dataset has been set to "Tensile Data 2055-2060 40C.xlsx".
  
*** If you would like to run files other than the defaults, you will have to comment out the default and uncomment the dataset you would like to run. The variable "original_ranges" will also need to be updated to correspond to the chosen dataset.
  
2) Change the file paths to save the graphs to the desktop.

QUICK TIP: Use cntr+f and type "save plot" to find the instances to change in the functions file.
