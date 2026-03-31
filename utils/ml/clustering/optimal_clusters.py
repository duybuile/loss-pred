import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


def find_optimal_clusters_db(data, min_clusters=2, max_clusters=10, random_state=42):
    """
    Find the optimal number of clusters using the Davies-Bouldin score.

    Parameters:
    - data: The input data for clustering
    - min_clusters: The minimum number of clusters to consider
    - max_clusters: The maximum number of clusters to consider
    - random_state: The random state to use

    Returns:
    - optimal_clusters: The optimal number of clusters
    """
    db_scores = []

    for num_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
        cluster_labels = kmeans.fit_predict(data)
        db_score = davies_bouldin_score(data, cluster_labels)
        db_scores.append(db_score)

    optimal_clusters_db = np.argmin(db_scores) + min_clusters
    return optimal_clusters_db


def find_optimal_clusters_silhouette(data, min_clusters=2, max_clusters=10, random_state=42):
    """
    Find the optimal number of clusters using the silhouette score.

    Parameters:
    - data: The input data for clustering
    - min_clusters: The minimum number of clusters to consider
    - max_clusters: The maximum number of clusters to consider
    - random_state: The random state to use

    Returns:
    - optimal_clusters: The optimal number of clusters
    """
    silhouette_scores = []

    for num_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Add min_clusters because the loop starts from there
    optimal_clusters = np.argmax(silhouette_scores) + min_clusters
    return optimal_clusters
