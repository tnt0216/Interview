"""Imported packages"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

"""Loading excel file"""
data = "Truncation Graphs.xlsx"

"""Creating an ExcelFile object to read the Excel file"""
xls = pd.ExcelFile(data)

"""Getting a list of all the sheet names (multiple sheets in the excel file)"""
sheet_names = xls.sheet_names

"""Creating empty lists to store individual data DataFrames and sample names"""
distances = []
radii = []

for index, sheet_name in enumerate(sheet_names):

    df = pd.read_excel(xls, sheet_name=sheet_name)
    print(df)

    """Placing all the values for distance and radii into the DataFrames"""
    df = df.sort_values(by=["Polymer"])
    print(df)

    df['Polymer'] = df['Polymer'].astype(str)

    # Filtering out polymer '30035' and reorder the remaining polymers
    ordered_polymers = ['2059', '2058', '2055', '2060', '2056', '2057']
    df_filtered = df[df['Polymer'].isin(ordered_polymers)]
    df_filtered['Polymer'] = pd.Categorical(df_filtered['Polymer'], categories=ordered_polymers, ordered=True)
    df_filtered = df_filtered.sort_values(['Polymer', 'Frequency'])

    # For low truncation I want to have 0.0 rad/s first for high I want 100 rad/s first
    if index > 1:
        df_filtered = df_filtered.sort_values(['Polymer', 'Frequency'], ascending=[True, False])

    print(df_filtered)

    # Setting the positions for each group of bars
    bar_width = 0.2  # Width of each bar
    x_positions = np.arange(len(ordered_polymers))  # Base positions for each polymer

    # Getting unique frequencies
    frequencies = df_filtered['Frequency'].unique()

    # Defining custom colors for the histograms
    colors = ['paleturquoise', 'navajowhite', 'mediumspringgreen', 'violet']  # You can adjust these colors

    print(df_filtered)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plotting bars for each frequency, shifting the x_positions for each one
    for i, freq in enumerate(frequencies):
        # Filter the DataFrame for the current frequency
        freq_data = df_filtered[df_filtered['Frequency'] == freq]

        # Aligning the x positions for the polymer, offset by the index of the frequency
        ax.bar(x_positions + i * bar_width, freq_data['Distance from 30035'],
               bar_width, label=f'{freq} rad/s', yerr=freq_data['Radii'],
               capsize=5, color=colors[i % len(colors)])  # Assign color to each bar

    # Labels and title
    ax.set_xlabel('Polymer', fontsize=12)
    ax.set_ylabel('Scaled Distance from Lot 30035', fontsize=12)
    # ax.set_title('Distance from 30035 for Different Polymers and Frequencies', fontsize=14)

    # Setting x-axis ticks to be in the middle of the grouped bars
    ax.set_xticks(x_positions + bar_width * (len(frequencies) - 1) / 2)
    ax.set_xticklabels(ordered_polymers)

    # Y-axis formatting
    ax.tick_params(axis='y', labelsize=12)

    # Adding a legend
    ax.legend(title='Cutoff Frequency')

    plt.tight_layout()
    plt.show()
