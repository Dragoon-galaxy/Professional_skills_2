import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate random data
X, y = make_blobs(n_samples=1000, centers=3, random_state=42)

# Create k-means clustering object
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)

# Fit the data to the model
kmeans.fit(X)

# Get the predicted labels and cluster centers
labels = kmeans.predict(X)
centers = kmeans.cluster_centers_

# Plot the data with different colors for each cluster
plt.scatter(X[:, 0], X[:, 1], c=labels)
# Plot the cluster centers as black dots
plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x')
plt.show()
