# HIERARCHICAL CLUSTERING

# two types --> Agglomerative(bottom up, more common) and Divisive(top down)

# Agglomerative Clustering
# 1) make each data point a single point cluster --> that forms n clusters
# 2) Take two closest data points and make them one cluster --> that forms n-1 clusters
# 3) Take two closest clusters and make them one cluster --> that forms n-2 clusters
# 4) repeat step 3 until there is only one cluster
# hierarchical clustering maintains the memory of how it went the whole process and that 
# memory is stored in dendograms

# dendograms contain data points on x-axis and euclidean distance on y-axis 
# and then in a bottom up fashion it draws the connecting clusters 
# with y distance representing dissimilarity(distance)

# Now to have required number of cluster
# we can set a threshold(dissimilarity,y-axis) and get the required number of clusters
# number of clusters equals the threshold line crossing vertical dendograms line
# Now to have optimal number of clusters we find 
# the largest vertical line that does not cross any extended horizontal line 

# for step 3 we need to calculate closest clusters
# 1) closest point
# 2) furthest point
# 3) average distance
# 4) distance between centroids

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
# y = dataset.iloc[:, 3].values

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
# ward is similar to minimizing wcss, actually it minimizes within cluster variance
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()