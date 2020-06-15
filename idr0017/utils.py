import os
from os.path import join, exists
import pickle
import json
import numpy as np
from scipy.spatial.distance import squareform

import umap
import hdbscan
from fastcluster import linkage

DATA_FOLDER = "data"
UMAP_FOLDER = "umap"
HDBSCAN_FOLDER = "hdbscan"


def get_umap_projection(data_ftr_vals, n_components=2, seed=4285866):
    """
    Calculate the UMAP projection for a feature data matrix with caching.
    """
    if not exists(join(DATA_FOLDER, UMAP_FOLDER)):
        os.makedirs(join(DATA_FOLDER, UMAP_FOLDER))
    fname = ("umap_" + str(n_components) + "_" + str(seed) + "_" +
             str(data_ftr_vals[0, ...].sum()) + ".pkl")

    if exists(join(DATA_FOLDER, UMAP_FOLDER, fname)):
        print("Loading cached UMAP...")
        with open(join(DATA_FOLDER, UMAP_FOLDER, fname), "rb") as f:
            umap_data = pickle.load(f)
            return umap_data["embedding"], umap_data["reducer"]

    np.random.seed(seed)
    reducer = umap.UMAP(n_components=n_components, random_state=seed)
    embedding = reducer.fit_transform(data_ftr_vals)
    with open(join(DATA_FOLDER, UMAP_FOLDER, fname), "wb") as f:
        pickle.dump({"embedding": embedding, "reducer": reducer}, f)
    return embedding, reducer


def get_hdbscan_clustering(embedding, min_cluster_size=60, cluster_selection_epsilon=0.32,  min_samples=30):
    if not exists(join(DATA_FOLDER, HDBSCAN_FOLDER)):
        os.makedirs(join(DATA_FOLDER, HDBSCAN_FOLDER))
    fname = ("hdbscan_" + str(min_cluster_size) +
             "_" + str(cluster_selection_epsilon) +
             "_" + str(min_samples) +
             "_" + str(embedding[0, ...].sum()) + ".pkl")

    if exists(join(DATA_FOLDER, HDBSCAN_FOLDER, fname)):
        print("Loading cached HDBSCAN...")
        with open(join(DATA_FOLDER, HDBSCAN_FOLDER, fname), "rb") as f:
            hdbscan_data = pickle.load(f)
            return hdbscan_data["labels"], hdbscan_data["clusterer"]

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                cluster_selection_epsilon=cluster_selection_epsilon, min_samples=min_samples).fit(embedding)
    hdbscan_data = {"labels": clusterer.labels_,  "clusterer": clusterer}
    with open(join(DATA_FOLDER, HDBSCAN_FOLDER, fname), "wb") as f:
        pickle.dump(hdbscan_data, f)
    return hdbscan_data["labels"], hdbscan_data["clusterer"]


# Unsupervised feature selection
def rank_features(data_matrix, criterion="variance"):
    sorted_idxs = []
    if criterion == "variance":
        # Highest variance
        variances = np.var(data_matrix, axis=0)
        sorted_idxs = sorted(range(len(variances)),
                             key=lambda i: variances[i], reverse=True)
    elif criterion == "normalized_std":
        # Highest std/max
        norm_std_devs = np.divide(
            np.std(data_matrix, axis=0), np.max(data_matrix, axis=0))
        sorted_idxs = sorted(range(len(norm_std_devs)),
                             key=lambda i: norm_std_devs[i], reverse=True)
    else:
        raise ValueError("Select a valid criterion")
    return sorted_idxs
