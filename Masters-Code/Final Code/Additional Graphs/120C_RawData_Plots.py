"""Importing packages"""
import matplotlib.pyplot as plt
import pandas as pd


def makeplot(sample_names, ranges, freq_data, data, namedata):

    """Commented out one is for single polymer"""
    marker_styles = ['o', 's', '^', 'v', 'D', '>', 'p']
    colors = ['#BCBD22', '#1F77B4', '#FF7F0E', '#2CA02C', '#9467BD', '#8C564B', '#7F7F7F']

    polymer_colors = []
    test_styles = []

    # Plot for each test in the lot
    plt.figure(figsize=(10, 6))
    plt.title(f"{namedata} Data for Polymer Lots (120°C)")
    plt.xlabel("Frequency [Hz]")
    if namedata == "Viscosity":
        plt.ylabel(f"{namedata} [Pa·s]")
    else:
        plt.ylabel(f"{namedata} [Pa]")

    # Plotting for each polymer lot
    for lot_index, (start, end) in enumerate(ranges):
        lot_color = colors[lot_index]
        lot_marker = marker_styles[lot_index]
        lot_name = sample_names[lot_index]

        for test_index in range(start, end + 1):
            test_freq = freq_data.iloc[test_index].dropna().tolist()  # Handle missing values
            test_data = data.iloc[test_index].dropna().tolist()
            plt.plot(
                test_freq,
                test_data,
                color=lot_color,
                marker=lot_marker,
                label=f"Lot {lot_name}" if test_index == start else "",
                linewidth=0.5,
                markersize=3
            )

    plt.legend(title="Polymer Lots", loc='best')
    plt.tight_layout()
    plt.show()


def main():

    """Loading excel file"""
    data = "2055-2060 Series 120C Batch 2 Outliers.xlsx"

    """Creating an ExcelFile object to read the Excel file"""
    xls = pd.ExcelFile(data)

    """Getting a list of all the sheet names (multiple sheets in the excel file)"""
    sheet_names = xls.sheet_names

    """Creating empty lists to store various data"""
    freq_row = []
    storage_row = []
    loss_row = []
    vis_row = []

    for index, sheet_name in enumerate(sheet_names):

        if index == 0:  # Skip the first sheet
            print("Ed's Calculations: ")
            eds_data = pd.read_excel(xls, sheet_name=sheet_name, usecols="C:E")
            print(eds_data)
            continue

        df = pd.read_excel(xls, sheet_name=sheet_name, usecols="D:G")
        freq = df.iloc[:, 0].tolist()
        storage = df.iloc[:, 1].tolist()
        loss = df.iloc[:, 2].tolist()
        vis = df.iloc[:, 3].tolist()

        freq_row.append(freq)
        storage_row.append(storage)
        loss_row.append(loss)
        vis_row.append(vis)

    freq_data = pd.DataFrame(freq_row)
    storage_data = pd.DataFrame(storage_row)
    loss_data = pd.DataFrame(loss_row)
    vis_data = pd.DataFrame(vis_row)

    print(freq_data)
    print(storage_data)
    print(loss_data)
    print(vis_data)

    ranges = [(0, 5), (6, 11), (12, 17), (18, 22), (23, 28), (29, 33), (34, 39)]  # Outliers Dataset 120C
    sample_names = ["2055", "2056", "2057", "2058", "2059", "2060", "30035"]

    namedata1 = "Storage"
    namedata2 = "Loss"
    namedata3 = "Viscosity"

    makeplot(sample_names, ranges, freq_data, storage_data, namedata1)
    makeplot(sample_names, ranges, freq_data, loss_data, namedata2)
    makeplot(sample_names, ranges, freq_data, vis_data, namedata3)


if __name__ == "__main__":
    main()
