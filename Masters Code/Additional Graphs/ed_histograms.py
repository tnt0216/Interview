import pandas as pd
from matplotlib import pyplot as plt

# Reading in Ed's histogram calculations from Excel file
dataset = "eds_histogram_calculation.xlsx"
xls = pd. ExcelFile(dataset)
eds_data = pd.read_excel(xls, usecols="A:M")
print(eds_data)

# Converting viscosity values to KPa*s
viscosity = (eds_data.iloc[:, 1]) / 1000
viscosity_err = (eds_data.iloc[:, 2]) / 1000

# Converting crossover modulus to KPa
crossover = eds_data.iloc[:, 9] / 1000
crossover_err = eds_data.iloc[:, 10] / 1000

avg_names = ["30035", "2055", "2056", "2057", "2058", "2059", "2060"]

# Creating histograms for viscosity, crossover modulus, and crossover frequency
fig, ax = plt.subplots(figsize=(3.75, 3.75))
ax.bar(range(len(avg_names)), viscosity, yerr=viscosity_err, capsize=5, color='paleturquoise')
# ax.set_title(f"Zero-Shear Viscosity Histogram", fontsize=12)
ax.set_ylabel('$\eta_0$ (KPa*s)', fontsize=12)
ax.set_xlabel(f"Polymer", fontsize=12)
ax.set_xticks(range(len(avg_names)))
ax.set_xticklabels(avg_names, rotation=90, fontsize=12)
ax.set_yticklabels(ax.get_yticks(), fontsize=12)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(3.75, 3.75))
ax.bar(range(len(avg_names)), eds_data.iloc[:, 5], yerr=eds_data.iloc[:, 6], capsize=5, color='paleturquoise')
# ax.set_title(f"Crossover Frequency Histogram", fontsize=16)
ax.set_ylabel("$\omega_{c}$ (rad/s)", fontsize=12)
ax.set_xlabel(f"Polymer", fontsize=12)
ax.set_xticks(range(len(avg_names)))
ax.set_xticklabels(avg_names, rotation=90, fontsize=12)
ax.set_yticklabels(ax.get_yticks(), fontsize=12)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(3.75, 3.75))
ax.bar(range(len(avg_names)), crossover, yerr=crossover_err, capsize=5, color='paleturquoise')
# ax.set_title(f"Crossover Modulus Histogram", fontsize=12)
ax.set_ylabel("$G_{c}$ (KPa)", fontsize=12)
ax.set_xlabel(f"Polymer", fontsize=12)
ax.set_xticks(range(len(avg_names)))
ax.set_xticklabels(avg_names, rotation=90, fontsize=12)
ax.set_yticklabels(ax.get_yticks(), fontsize=12)
plt.tight_layout()
plt.show()
