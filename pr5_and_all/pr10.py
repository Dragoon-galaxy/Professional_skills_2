from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize KMeans with 3 clusters (since there are 3 classes in the Iris dataset)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # Explicitly setting n_init

# Fit KMeans to the scaled data
kmeans.fit(X_scaled)

# Get cluster centers and labels
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Plotting the clusters
plt.figure(figsize=(10, 6))

plt.scatter(X_scaled[labels == 0, 0], X_scaled[labels == 0, 1], s=50, c='red', label='Cluster 1')
plt.scatter(X_scaled[labels == 1, 0], X_scaled[labels == 1, 1], s=50, c='blue', label='Cluster 2')
plt.scatter(X_scaled[labels == 2, 0], X_scaled[labels == 2, 1], s=50, c='green', label='Cluster 3')

plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=200, c='black', marker='X', label='Centroids')

plt.title('K-means Clustering of Iris Dataset')
plt.xlabel('Sepal Length (scaled)')
plt.ylabel('Sepal Width (scaled)')
plt.legend()
plt.show()
