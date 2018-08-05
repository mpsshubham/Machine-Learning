# K-MEANS CLUSTERING

# Clustering is similar to classification, but the basis is different.
# In Clustering you don’t know what you are looking for, and you are trying to identify
# some segments or clusters in your data. When you use clustering algorithms on your dataset,
# unexpected things can suddenly pop up like structures, clusters and groupings 
# you would have never thought of otherwise.

# 1) choose the number k of clusters
# 2) select at random k points,the centroids(not necessarily from your dataset)
# 3) assign each data point to the closest centroid --> that forms k clusters
# 4) compute and place the new centroid of each cluster
# 5) reassign each data point to the new closest centroid 
#    (if any reassignment took place, goto step4, otherwise FIN)

# Random Initialization Trap
# Sometimes selection of centroid at the start can dictate the outcome of the algorithm
# the solution of the problem is kmeans++ which helps to choose good starting centroids

# Choosing the right number of clusters
# we need to have certain metric that evaluates performance of different number of clusters
# WCSS --> within cluster sum of squares also called inertia
# WCSS = summation of all Pi in cluster 1 (distance(Pi,C1)^2) + summation of all Pi in cluster 2 (distance(Pi,C2)^2)
# this WCSS metric always decrease with increase in the number of clusters
# as the distance of the points from clusters will eventually decrease with increase in clusters
# at max we can have clusters equal to number of points, in that case WCSS will be zero 
# so to overcome this problem we use elbow method
# so we look for that change(elbow) where drop goes from substantial to less substanial(high to low drop)


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
# y = dataset.iloc[:, 3].values

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42) #init can be random 
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()