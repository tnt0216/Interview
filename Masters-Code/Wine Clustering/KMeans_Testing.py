# Taylor Tallerday
# This program is intended to run a Python k-means script for testing it works on two-dimensional data
# that is hardcoded using arrays
# It uses the elbow method to determine the optimal number of clusters for a dataset

# Run the base case with wine data to see the differences between the two methods

# Importing modules
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

# Reading in the wine data from the excel speadsheet
data_df = pd.read_csv('wine-clustering.csv')

# Reading in the 13 dimensions
dimension1 = data_df['Alcohol']
dimension2 = data_df['Malic_Acid']
dimension3 = data_df['Ash']
dimension4 = data_df['Ash_Alcanity']
dimension5 = data_df['Magnesium']
dimension6 = data_df['Total_Phenols']
dimension7 = data_df['Flavanoids']
dimension8 = data_df['Nonflavanoid_Phenols']
dimension9 = data_df['Proanthocyanins']
dimension10 = data_df['Color_Intensity']
dimension11 = data_df['Hue']
dimension12 = data_df['OD280']
dimension13 = data_df['Proline']


# Plotting first and second dimensions on scatter plot
plt.scatter(dimension1, dimension2)
plt.xlabel('Alcohol')
plt.ylabel('Malic_Acid')
plt.title('Scatter Plot')
plt.show()

# Converting data to a list of tuples for clustering
data = data_df[['Alcohol', 'Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium', 'Total_Phenols', 'Flavanoids', 'Nonflavanoid_Phenols', 'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline']].values.tolist()

# Determining the best value for K be training KMeans model
# inertia is the sum of the squared distances of samples to their closest cluster center
inertias = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    # This tells us how far away the points within a cluster are (small inertias are aimed for)
    # The range of inertias' value starts from zero and goes up
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 11), inertias, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Based on the previous graph we determined which value was determined for K
# This was determined to be 2 for this example dataset so 2 clusters will be used
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

plt.scatter(dimension1, dimension2, c=kmeans.labels_)
plt.xlabel('Alcohol')
plt.ylabel('Malic_Acid')
plt.title('Clustered Data')
plt.show()
