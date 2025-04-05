*************************
Master's Research Project
************************


PROJECT DESCRIPTION:
--------------------
The purpose of this project is to assess the rheological and tensile similarity between various PVDF-CTFE legacy lots with varying synthesis and processing techniques. Dimension reduction (PCA / LDA) was used to reduce datasets containing 100+ dimensions to 2 dimensions to qualitatively assess similarity. Distance calculations were also made to quantitatively assess similarity. Clustering (k-means) was also applied to group similar polymer lots together to further verify similarities and assess outliers.

VERSION REQUIREMENTS:
--------------------
These scripts were written using Python 3.9.13.

HOW TO USE:
----------
In the Masters-Code folder, there is a folder named "Final Code" that contains subfolders titled "Additional Graphs", "Excel Files", and "Mains". 

Additional Graphs: Contains a few short scripts for additional graphs we wanted to use in our papers.
  
Excel Files: Contains all of the final data used for our scientific journal and my thesis. Subfolders are separating the rheological and tensile datasets. The first three Excel files were focused on in our analysis and are referenced in the rheology_dataset_main. We only collected one full high-res dataset, which was referenced in the high_res_dataset_main. Finally, the tensile datasets were referenced in the tensile_dataset_main.

Mains: This folder contains the main functions/main scripts.

Functions: This file contains all the various functions that each of the main scripts executes. 

*NOTES:
When running the code file paths will have to be changed to save various graphs correctly. To quickly search and change these paths cntr f "save_plot" within the functions file.

