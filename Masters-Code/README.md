Master's Research Project
=========================

PROJECT DESCRIPTION:
--------------------
The purpose of this project is to assess the rheological and tensile similarity between various PVDF-CTFE legacy lots with varying synthesis and processing techniques. Dimension reduction (PCA / LDA) was used to reduce datasets containing 100+ dimensions to 2 dimensions to qualitatively assess similarity. Distance calculations were also made to quantitatively assess similarity. Clustering (k-means) was also applied to group similar polymer lots together to further verify similarities and assess outliers. Only the final versions of code are contained within this repo for viewing.

VERSION REQUIREMENTS:
--------------------
These scripts were written using Python 3.9.13.

NAVIGATING FOLDERS:
-------------------
In the Masters-Code folder, there is a folder named "Final Code" that contains subfolders titled "Additional Graphs", "Excel Files", and "Mains". 

Additional Graphs: Contains a few short scripts for additional graphs we wanted to use in our papers.
  
Excel Files: Contains all of the final data used for our scientific journal and my thesis. Subfolders are separating the rheological and tensile datasets. The first three Excel files were focused on in our analysis and are referenced in the rheology_dataset_main. We collected one full high-res dataset, which was referenced in the high_res_dataset_main to investigate the importance of high-frequency data. Finally, the tensile datasets were referenced in the tensile_dataset_main.

Mains: This folder contains the main functions/main scripts.

Functions: This file contains all the various functions that each of the main scripts executes. 

HOW TO USE:
---------------
1) Download appropriate files: the function file (used by each main), the main you would like to run (rheology_dataset_main, high_res_rheology_main, or tensile_dataset_main), and the dataset/Excel file.

Notes:
- When running "rheology_dataset_main": The default dataset has been set to "2055-2060 Series 120C Batch 2 Outliers.xlsx".
- When running "high_res_rheology_main": The corresponding dataset is titled "High Precision 2055-2060 Series 170C.xlsx".
- When running "tensile_dataset_main": The default dataset has been set to "Tensile Data 2055-2060 40C.xlsx".
  
*** If you would like to run files other than the defaults, you will have to comment out the default and uncomment the dataset you would like          to run. The variable "original_ranges" will also need to be updated to correspond to the chosen dataset.
  
3) Change the file paths for saving the graphs to the desktop.

QUICK TIP: Use cntr f "save plot" to find the instances to change in the functions file.
