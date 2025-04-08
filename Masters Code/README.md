Master's Research Project
=========================

PROJECT DESCRIPTION:
--------------------
The purpose of this research was to assess the rheological and tensile similarity between various PVDF-CTFE legacy lots with varying synthesis and processing techniques. Dimension reduction (PCA / LDA) was used to reduce datasets containing 4 or more dimensions to 2 dimensions to qualitatively assess similarity. Distance calculations were also made to quantitatively assess similarity. Clustering (k-means) was applied to group similar polymer lots together to further verify similarities and assess outliers. Only the final versions of code are contained within this repo for viewing.

VERSION REQUIREMENTS:
--------------------
These scripts were written using Python 3.9.13. I have included the requirements.txt for the environment set up with pip.

NAVIGATING FOLDERS:
-------------------
In the "Masters Code" folder, there are subfolders titled "Additional Graphs", "Excel Files", and "Mains". 

Additional Graphs: Contains a few short scripts for additional graphs we wanted to use in our papers.
  
Excel Files: Contains all of the final data used for our scientific journal and my thesis. Subfolders are separating the rheological and tensile datasets.

About the Datasets:

**Rheology - rheology_dataset_main**
- The rheology datasets were the primary datasets used in our analysis and were the focus of the research journal and thesis. There were three rheology datasets for temperatures of 70°C, 120°C, and 170°C used by "rheology_dataset_main". Each of these datasets had undergone data preprocessing by applying Chavunet's Criteria to identify outliers in the dataset. These outliers were taken completely out before running the data through these scripts. We compared our results to an experimental approach of assessing rheological similarity through the crossover point and zero-shear viscosity. These outliers were not included and handled in this code because we wanted to have identical datasets to the experimental approach for direct comparisons of performance.
  - Original Dimenions 70°C rheology dataset = 39 (13 datapoints collected x3 for storage, loss, viscosity)
  - Original Dimenions 120°C rheology dataset = 57 (19 datapoints collected x3 for storage, loss, viscosity)
  - Original Dimenions 170°C rheology dataset = 57 (19 datapoints collected x3 for storage, loss, viscosity)

**High-Res Rheology - high_res_rheology_main**
- We investigated the impact of high-frequency data on our results to determine if it was worthwhile to perform larger frequency sweeps. We determined that it was not worthwhile to pursue, however, the code and high-res dataset has been included and can be ran through the "high_res_rheology_main" using the "High Precision 2055-2060 Series 170C.xlsx" dataset.
  -  Original Dimenions high-res 170°C rheology dataset = 150 (50 datapoints collected x3 for storage, loss, viscosity)

*In addition to these rheology datasets I have included Excel files within the Rheology folder that were used to create truncation graphs, and Ed's histograms (experimental approach to assessing similarity). To run these Excel files you will need to run the mains located in the Additional Graphs folder*

  **Tensile - tensile_dataset_main**
- Four temperature tensile datasets were collected for analysis, which were 24°C, 40°C, 50°C, and 70°C, respectively. These datasets had to be adjusted in preprocessing so that they did not contain negative values, each test contained the same number of sampled datapoints/dimensions, and each of the datapoints being compared to each other corresponded to the same strain value, the first dimension/first value in each test was set to zero and following datapoints were shifted accordingly.
  - Original Dimensions of 24°C, 40°C, 50°C, 70°C tensile dataset is dependent on how many interpolated points are chosen.

Mains: This folder contains the main functions/main scripts.

Functions: This file contains all the various functions that each of the main scripts executes. 

HOW TO USE:
---------------
1) Create a clone of this repo
2) Make sure that the clone is saved to a file where your virtual environment can assess the clone
3) Open the functions file and the main that you would like to run
4) Run the main
5) Provide inputs to the questions printed in the command line while running

**All outputs should autopopulate into the "Outputs" folder that is created when you run each main**
