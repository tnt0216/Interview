Lockheed Martin Constellation Tool
==================================

PROJECT DESCRIPTION:
--------------------
The purpose of this project is to implement a tool to allow more streamlined data calculation for astrodynamics related systems.

This application allows a user to enter site information manually or input a certain file format to output COE information, and given a time of flight, output new sight information.

The UI utilizes MATLAB, and the team attempted to utilize Python. However, the MATLAB engine module would not work correctly. Furthermore, a visual ground tracking window was going to be implemented, but was not able to be completed within the time constraint. 

VERSION REQUIREMENTS:
--------------------
The code was written in MATLAB R2023a, and this version of MATLAB (or later) may be required to run the software correctly. Alternatively, an older version of MATLAB, such as version R2022b has been used to open the UI, and the UI automatically updated itself to work with this version; however, R2023a may still be required to be installed for this function to work.

The following website will direct you to a download site for MATLAB 2023a:
https://www.mathworks.com/downloads/?s_tid=rh_bn_dl

To open the UI, either double-click on the "Manual_UI.mlapp" tool and select "open with" MATLAB R2023a. Or whichever compatible version of MATLAB is installed. Double left-clicking the "Manual_UI.mlapp" tool will open the application on the most current version installed or the default version of MATLAB. Sometimes the GUI will not open when double left clicking, and the user will need to run the "Manual_UI.mlapp" file from MATLAB's "Current Folder" area.

NAVIGATING FOLDERS:
------------------
Data: This folder contains example inputs for testing and outputs.

Backend: This folder contains the separate backend astrodynamics functions used by the main function. (written by Cole Stoumbaugh and Taylor Tallerday (ME)) - These functions have been further categorized to indicate which functions I wrote and which were written by Cole. A MATLAB file containing the full working backend code has also been included.

Lockheed Code: This folder contains pre-existing functions written by Lockheed Martin employees.

UI: This folder contains the functions for the user interface (written by Matt Terry and Taylor Phair (NOT ME))

full_backend_code.m: The final MATLAB file that contains all the functions including the main.

HOW TO USE:
-----------

UI NOTES:
The "Manual Input" tab allows the user to input site information in km, degrees, and seconds. When the user selects "Run", the manual output window will be populated with COE information, and new visible site locations as well as all sites available.

The "Manual Output" tab is where this information is output. The user has the option to save this information through a "browse" function. The "*txt" will need to be changed to "All Files" and the name of the file will need a readable extension such as ".txt".

The "File Input" tab allows the user to search for a specific formatted file. This does work with .ORBSET files as well. Alternatively, the user can directly input the file path into the "File Input" box. Once the user hits "Calculate," the window will populate with COEs. The optional TOF box is used to find a new visible site location. The user, again, has the option to save this data.

The "Orbit Maneuver" tab will require the Manual Input and File Output tab, but will not work for both at the same time. This tab requires a transfer orbit radius to be input in kilometers. Depending on what type of transfer orbit is used, an optimal delta-v and optimal TOF will be displayed. TOF is displayed in hours. Delta V is displayed in kilometers per second, wait time is displayed in hours, and the optimal transfer radius is displayed in kilometers.

CREDITS:
-------
This project was created for a University of Colorado Colorado Springs senior design project. The following team members contributed to this code:
- Stoumbaugh, Cole (Team Lead)
- Coffield, Thomas (Communications Lead)
- Tallerday, Taylor (Computer Science - Back End Programming)
- Terry, Matt (Computer Science - User Interface Programming)
- Phair, Taylor (Computer Science - User Interface Design)
