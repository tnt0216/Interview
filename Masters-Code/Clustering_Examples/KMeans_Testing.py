# Taylor Tallerday
# This program is intended to run a Python k-means script for testing it works on two-dimensional data
# that is hardcoded using arrays

# Run the base case with wine data to see the differences between the two methods

# Importing modules
import csv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#filename = "Financials Sample Data.csv"

# Creating arrays for the datasets
x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

plt.scatter(x, y)
plt.show()

data = list(zip(x, y))
print(data)


# Determining the best value for K be training KMeans model
inertias = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    # Dig into how this works
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 11), inertias, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Based on the previous graph we determined which value was determined for K
# This was determined to be 2 for this example dataset so 2 clusters will be used

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

plt.scatter(x, y, c=kmeans.labels_)
plt.show()
