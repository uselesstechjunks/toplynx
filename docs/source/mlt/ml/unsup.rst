###################################################################################
Unsupervised Learning
###################################################################################

***********************************************************************************
Dimensionality Reduction
***********************************************************************************
Geometric
===================================================================================
Variational
===================================================================================
Evaluation Criteria
===================================================================================

***********************************************************************************
Clustering
***********************************************************************************
Metric Based
===================================================================================
K-Means
-----------------------------------------------------------------------------------
.. code-block:: python

	class KMeans:
	    def __init__(self, n_clusters, max_iter=100):
	        self.n_clusters = n_clusters
	        self.max_iter = max_iter
	        self.centroids = None
	
	    def fit(self, X):
	        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
	        for _ in range(self.max_iter):
	            clusters = [[] for _ in range(self.n_clusters)]
	            for x in X:
	                distances = np.linalg.norm(self.centroids - x, axis=1)
	                cluster_idx = np.argmin(distances)
	                clusters[cluster_idx].append(x)
	            new_centroids = [np.mean(cluster, axis=0) for cluster in clusters]
	            if np.allclose(self.centroids, new_centroids):
	                break
	            self.centroids = np.asarray(new_centroids)
	
	    def predict(self, X):
	        distances = np.linalg.norm(self.centroids[:, np.newaxis] - X, axis=2)
	        return np.argmin(distances, axis=0)

Spectral Clustering
-----------------------------------------------------------------------------------
.. code-block:: python

	class SpectralClustering:
	    def __init__(self, n_clusters, n_neighbors=10, sigma=1.0):
	        self.n_clusters = n_clusters
	        self.n_neighbors = n_neighbors
	        self.sigma = sigma
	        self.labels_ = None
	    
	    def fit_predict(self, X):
	        # Step 1: Construct the k-nearest neighbors graph
	        W = self._construct_knn_graph(X)
	        
	        # Step 2: Compute the graph Laplacian (normalized)
	        L = self._compute_laplacian(W)
	        
	        # Step 3: Perform eigen decomposition
	        eigenvalues, eigenvectors = np.linalg.eigh(L)
	        
	        # Step 4: Extract the eigenvectors corresponding to the smallest eigenvalues
	        embedding = eigenvectors[:, 1:self.n_clusters + 1]  # Exclude the first eigenvalue/eigenvector
	        
	        # Step 5: Cluster the embedded points using K-means
	        from sklearn.cluster import KMeans
	        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
	        self.labels_ = kmeans.fit_predict(embedding)
	        
	        return self.labels_
	    
	    def _construct_knn_graph(self, X):
	        n_samples = X.shape[0]
	        W = np.zeros((n_samples, n_samples))
	        
	        for i in range(n_samples):
	            dists = np.linalg.norm(X - X[i], axis=1)
	            idx = np.argsort(dists)[:self.n_neighbors]
	            W[i, idx] = np.exp(-dists[idx] ** 2 / (2 * self.sigma ** 2))
	        
	        return W
	    
	    def _compute_laplacian(self, W):
	        D = np.diag(np.sum(W, axis=1))  # Degree matrix
	        L = D - W  # Unnormalized Laplacian
	        D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(D) + 1e-8))  # Avoid division by zero
	        L_normalized = np.dot(np.dot(D_inv_sqrt, L), D_inv_sqrt)  # Normalized Laplacian
	        
	        return L_normalized

Density Based
===================================================================================
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
-----------------------------------------------------------------------------------
.. code-block:: python

	class DBSCAN:
	    def __init__(self, eps=0.5, min_samples=5):
	        self.eps = eps
	        self.min_samples = min_samples
	        self.labels = None
	        self.visited = None
	        self.core_samples = None
	        self.n_clusters = None
	
	    def fit_predict(self, X):
	        self.labels = np.full(X.shape[0], -1)  # -1 indicates unclassified
	        self.visited = np.zeros(X.shape[0], dtype=bool)
	        self.core_samples = np.zeros(X.shape[0], dtype=bool)
	        self.n_clusters = 0
	
	        # Find core samples and assign cluster labels
	        for i in range(X.shape[0]):
	            if self.visited[i]:
	                continue
	            self.visited[i] = True
	
	            neighbors = self._find_neighbors(X, i)
	            if len(neighbors) < self.min_samples:
	                continue
	
	            self.core_samples[i] = True
	            self.n_clusters += 1
	            self.labels[i] = self.n_clusters - 1
	
	            # Expand cluster
	            self._expand_cluster(X, neighbors, self.n_clusters - 1)
	
	        return self.labels
	
	    def _find_neighbors(self, X, idx):
	        distances = np.linalg.norm(X - X[idx], axis=1)
	        return np.where(distances <= self.eps)[0]
	
	    def _expand_cluster(self, X, neighbors, cluster_id):
	        for neighbor in neighbors:
	            if not self.visited[neighbor]:
	                self.visited[neighbor] = True
	                new_neighbors = self._find_neighbors(X, neighbor)
	                if len(new_neighbors) >= self.min_samples:
	                    self.core_samples[neighbor] = True
	                    neighbors = np.concatenate((neighbors, new_neighbors))
	
	            if self.labels[neighbor] == -1:
	                self.labels[neighbor] = cluster_id

Evaluation Criteria
===================================================================================
CH-Index
-----------------------------------------------------------------------------------
.. code-block:: python

	def calculate_ch_index(X, labels):
	    """
	    Calculate the Calinski-Harabasz index to evaluate K-means clustering.
	
	    Parameters:
	    - X: numpy array, shape (n_samples, n_features)
	        Data points to be clustered.
	    - labels: numpy array, shape (n_samples,)
	        Cluster labels assigned to each data point.
	
	    Returns:
	    - ch_index: float
	        The computed Calinski-Harabasz index.
	    Steps:
	
	    - Determine the number of clusters (n_clusters) based on the maximum label value.
	    - Compute cluster centers by calculating the mean of points within each cluster.
	    - Calculate the mean distance between all pairs of cluster centers (mean_center_distance).
	    - Compute the mean within-cluster scatter (mean_within_scatter), which is half of the sum of pairwise distances within each cluster.
	    - Compute the CH-index using the formula
	    """
	    n_clusters = np.max(labels) + 1
	    n_samples = X.shape[0]
	    cluster_centers = np.empty((n_clusters, X.shape[1]))
	
	    # Calculate cluster centers
	    for k in range(n_clusters):
	        cluster_centers[k] = np.mean(X[labels == k], axis=0)
	
	    # Compute the mean distance between cluster centers
	    mean_center_distance = np.mean(pairwise_distances(cluster_centers))
	
	    # Compute the mean within-cluster scatter
	    mean_within_scatter = 0.0
	    for k in range(n_clusters):
	        cluster_points = X[labels == k]
	        if len(cluster_points) > 0:
	            mean_within_scatter += np.sum(pairwise_distances(cluster_points)) / (2 * len(cluster_points))
	
	    # Compute CH-index
	    ch_index = mean_center_distance / mean_within_scatter * (n_samples - n_clusters) / (n_clusters - 1)
	
	    return ch_index

DB-Index
-----------------------------------------------------------------------------------
.. code-block:: python

	def calculate_db_index(X, labels):
	    """
	    Calculate the Davies-Bouldin index to evaluate K-means clustering.
	
	    Parameters:
	    - X: numpy array, shape (n_samples, n_features)
	        Data points to be clustered.
	    - labels: numpy array, shape (n_samples,)
	        Cluster labels assigned to each data point.
	
	    Returns:
	    - db_index: float
	        The computed Davies-Bouldin index.
	    """
	    n_clusters = np.max(labels) + 1
	    n_samples = X.shape[0]
	
	    # Calculate cluster centers
	    cluster_centers = np.empty((n_clusters, X.shape[1]))
	    for k in range(n_clusters):
	        cluster_centers[k] = np.mean(X[labels == k], axis=0)
	
	    # Compute pairwise cluster distances
	    cluster_distances = pairwise_distances(cluster_centers)
	
	    # Initialize the Davies-Bouldin index
	    db_index = 0.0
	
	    for i in range(n_clusters):
	        # Calculate average similarity for each cluster
	        similarity = np.zeros(n_clusters)
	        for j in range(n_clusters):
	            if i != j:
	                similarity[j] = (np.sum(pairwise_distances(X[labels == i], X[labels == j])) / 
	                                 (len(X[labels == i]) + len(X[labels == j])))
	        if np.sum(similarity) > 0:
	            db_index += np.max(similarity) / np.sum(similarity)
	
	    db_index /= n_clusters
	
	    return db_index

Silhoutte Coefficient
-----------------------------------------------------------------------------------
.. code-block:: python

	def calculate_silhouette_coefficient(X, labels):
	    """
	    Calculate the Silhouette Coefficient to evaluate K-means clustering.
	
	    Parameters:
	    - X: numpy array, shape (n_samples, n_features)
	        Data points to be clustered.
	    - labels: numpy array, shape (n_samples,)
	        Cluster labels assigned to each data point.
	
	    Returns:
	    - silhouette_avg: float
	        The computed average Silhouette Coefficient.
	    """
	    n_samples = X.shape[0]
	    cluster_labels = np.unique(labels)
	    n_clusters = len(cluster_labels)
	
	    if n_clusters == 1:
	        return 0.0  # Silhouette Coefficient is not defined for a single cluster
	
	    # Compute pairwise distances between samples
	    distances = pairwise_distances(X)
	
	    # Initialize arrays to store silhouette coefficients and cluster metrics
	    silhouette_values = np.zeros(n_samples)
	    cluster_means = np.zeros(n_clusters)
	
	    # Calculate mean distance of each sample to all other points in its cluster
	    for k in range(n_clusters):
	        cluster_points = X[labels == cluster_labels[k]]
	        cluster_size = len(cluster_points)
	        if cluster_size == 0:
	            cluster_means[k] = 0.0
	        else:
	            mean_distance = np.sum(distances[labels == cluster_labels[k]], axis=1) / cluster_size
	            cluster_means[k] = np.mean(mean_distance)
	
	    # Calculate silhouette coefficient for each sample
	    for i in range(n_samples):
	        curr_label = labels[i]
	        a_i = cluster_means[curr_label]  # Mean distance of i to other points in the same cluster
	
	        # Find the mean distance to points in the nearest neighboring cluster
	        b_i = np.inf
	        for k in range(n_clusters):
	            if k != curr_label:
	                mean_distance = np.mean(distances[i, labels == cluster_labels[k]])
	                if mean_distance < b_i:
	                    b_i = mean_distance
	
	        silhouette_values[i] = (b_i - a_i) / max(a_i, b_i)
	
	    # Average silhouette coefficient across all samples
	    silhouette_avg = np.mean(silhouette_values)
	    
	    return silhouette_avg

***********************************************************************************
Anomaly Detection
***********************************************************************************
Classifier Based
===================================================================================
Isolation Forest
-----------------------------------------------------------------------------------
Algorithm:

	- randomly assign a covariate to a note
	- randomly split between the [min, max] range of that covariate
	- stops when all the points are similar or just 1 point left within a region

Intuition: 

	- for isolated points, we'd need lesser number of cuts to isolate them to a region

Score:

	- average height of the tree for a given point, lower indicates outliers

One-Class SVM
-----------------------------------------------------------------------------------
Density Based
===================================================================================
Z-Score or Standard Score Method
-----------------------------------------------------------------------------------
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
-----------------------------------------------------------------------------------
Local Outlier Factor (LOF)
-----------------------------------------------------------------------------------
Evaluation Criteria
===================================================================================
