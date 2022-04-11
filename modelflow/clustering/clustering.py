import scipy.spatial.distance as spatial_distance
import pandas as pd
import numpy as np


def get_centroid_distance(kmeans):
    centroids = {count: i for count, i in enumerate(kmeans.cluster_centers_)}
    df = pd.DataFrame(columns=centroids.keys(), index=centroids.keys())
    for cluster1 in centroids.keys():
        for cluster2 in centroids.keys():
            df.loc[cluster1, cluster2] = spatial_distance.euclidean(
                centroids[cluster1], centroids[cluster2]
            )
    return df


def merge_clusters(labels, cluster_0, cluster_1):
    return [cluster_0 if i == cluster_1 else i for i in labels]


def similar_clusters(kmeans, similarity_cutoff):
    similarity = get_centroid_distance(kmeans)
    similarity = similarity.where(np.triu(np.ones(similarity.shape)).astype(np.bool))
    return (
        similarity[(similarity < similarity_cutoff) & (similarity != 0)]
        .stack()
        .index.tolist()
    )
