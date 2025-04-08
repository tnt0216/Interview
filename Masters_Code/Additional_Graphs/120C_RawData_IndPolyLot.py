import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def readindata(data):

    """This function reads in the data from the Excel file and inputs it all into specialized Pandas DataFrames"""

    xls = pd.ExcelFile(data)

    # Getting sheet names
    sheet_names = xls.sheet_names

    # Creating empty lists to store data
    freq_row = []
    storage_row = []
    loss_row = []
    vis_row = []

    # Readinging data from sheets
    for index, sheet_name in enumerate(sheet_names):
        if index == 0:  # Skip the first sheet
            print("Ed's Calculations: ")
            eds_data = pd.read_excel(xls, sheet_name=sheet_name, usecols="C:E")
            print(eds_data)
            continue

        # Reading specified columns from each sheet
        df = pd.read_excel(xls, sheet_name=sheet_name, usecols="D:G")
        freq_row.append(df.iloc[:, 0].tolist())
        storage_row.append(df.iloc[:, 1].tolist())
        loss_row.append(df.iloc[:, 2].tolist())
        vis_row.append(df.iloc[:, 3].tolist())

    # Converting data to DataFrames
    freq_data = pd.DataFrame(freq_row)
    storage_data = pd.DataFrame(storage_row)
    loss_data = pd.DataFrame(loss_row)
    vis_data = pd.DataFrame(vis_row)

    return freq_data, storage_data, loss_data, vis_data


def makingplots(freq_data, storage_data, loss_data, vis_data, sample_name):
    """
    This function makes the plots for individual polymer lots. Each plot includes the storage modulus,
    loss modulus, and viscosity data for all tests of a single polymer lot, with each test having a unique color.
    """

    # Generating a unique color for each test
    test_colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A8', '#A833FF', '#33FFF8', '#F8FF33']
    marker_styles = ['o', 's', '^', 'v', 'D', '>', 'p']  # Marker styles for visual distinction

    num_tests = len(vis_data)  # Number of tests (rows in the data)

    # Expanding the colors list if there are more tests than colors
    if num_tests > len(test_colors):
        test_colors = test_colors * ((num_tests // len(test_colors)) + 1)

    # Creating the figure and axes
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.set_title(f"{sample_name} Raw Data for 120°C Dataset", fontsize=12)
    ax.set_xlabel('Frequency [Hz]', fontsize=12)
    ax.set_ylabel("G' [Pa], G'' [Pa], |η*| [Pa·s]", fontsize=12)

    # Ensuring consistent frequency axis data
    freq_data_list = freq_data.iloc[0].tolist()

    # Plotting each test
    for idx, (vis_row, storage_row, loss_row) in enumerate(zip(vis_data.iterrows(),
                                                               storage_data.iterrows(),
                                                               loss_data.iterrows())):
        # Extracting data for this test
        vis_values = vis_row[1].values
        storage_values = storage_row[1].values
        loss_values = loss_row[1].values

        # Assigning a unique color and marker for each test
        color = test_colors[idx]
        # marker = marker_styles[idx % len(marker_styles)]

        # Plotting viscosity data
        ax.loglog(freq_data_list, vis_values, label=f"Test {idx+1}: Viscosity",
                color=color, marker=marker_styles[1], linestyle='--')

        # Plotting storage modulus
        ax.loglog(freq_data_list, storage_values, label=f"Test {idx+1}: Storage Modulus",
                color=color, marker=marker_styles[2], linestyle='-.')

        # Plotting loss modulus
        ax.loglog(freq_data_list, loss_values, label=f"Test {idx+1}: Loss Modulus",
                color=color, marker=marker_styles[3], linestyle=':')

    # Adding legend and formatting
    ax.legend(title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.tight_layout()
    plt.show()


def main():

    # Loading Excel file
    data = "2055-2060 Series 70C Batch 2 Outliers.xlsx"
    # data = "2055-2060 Series 120C Batch 2 Outliers.xlsx"
    # data = "2055-2060 Series 170C Batch 2 Outliers.xlsx"

    freq_data, storage_data, loss_data, vis_data = readindata(data)

    # Defining ranges and sample names
    ranges = [(0, 5), (6, 11), (12, 18), (19, 24), (25, 30), (31, 36), (37, 42)]  # Outliers Dataset 70C
    # ranges = [(0, 5), (6, 11), (12, 17), (18, 22), (23, 28), (29, 33), (34, 39)]  # Outliers Dataset 120C
    # ranges = [(0, 5), (6, 11), (12, 17), (18, 23), (24, 30), (31, 36), (37, 43)]  # Outliers Dataset 170C
    sample_names = ["2055", "2056", "2057", "2058", "2059", "2060", "30035"]

    # Looping through the data to only pass in the data for one polymer lot type (polymer lot types are defined by
    # ranges)
    for i, (start, end) in enumerate(ranges):

        # Assigning polymer lot name
        sample_name = sample_names[i]

        # Pulling the rows in the dataframe that belong to the specific polymer lot
        freq_data_polylot = freq_data.iloc[start:end+1, 0:13]
        storage_data_polylot = storage_data.iloc[start:end+1, 0:13]
        loss_data_polylot = loss_data.iloc[start:end+1, 0:13]
        vis_data_polylot = vis_data.iloc[start:end+1, 0:13]

        print(storage_data_polylot)

        # Calling to makeplots function to make the individual plots for the polymer lots
        # Passing in i to indicate the polymer lot to define the color of the line graphs
        makingplots(freq_data_polylot, storage_data_polylot, loss_data_polylot, vis_data_polylot, sample_name)


if __name__ == "__main__":
    main()
