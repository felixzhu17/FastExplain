import scipy.spatial.distance as spatial_distance
from ..utils import Utils
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as sch
import plotly.figure_factory as ff


def feature_correlation(xs, plotsize=(1000, 1000)):
    keep_cols = [i for i in xs.columns if len(xs[i].unique()) > 1]
    corr = np.round(spearmanr(xs[keep_cols]).correlation, 4)
    fig = ff.create_dendrogram(
        1 - corr,
        orientation="left",
        labels=xs.columns,
        distfun=squareform,
        linkagefun=lambda x: sch.linkage(x, "average"),
    )
    fig.update_layout(width=plotsize[0], height=plotsize[1], plot_bgcolor="white")
    return fig


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


class ClusteringClassified:
    def __init__(self, m, xs):
        self.m = m
        self.xs = xs

    def feature_correlation(self, *args, **kwargs):
        return feature_correlation(self.xs, *args, **kwargs)
