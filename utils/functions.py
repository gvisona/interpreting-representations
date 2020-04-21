import numpy as np
import umap

from scipy.spatial.distance import squareform
from collections import OrderedDict

from fastcluster import linkage

def get_umap_projection(data_ftr_vals, n_components=2, seed=4285866):
    np.random.seed(seed)
    reducer = umap.UMAP(n_components=n_components, random_state=seed)
    embedding = reducer.fit_transform(data_ftr_vals)
    return embedding, reducer



#######################################
# Reordering a matrix
# from https://gmarti.gitlab.io/ml/2017/09/07/how-to-sort-distance-matrix.html

def seriation(Z,N,cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z
            
        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))
    
def compute_serial_matrix(dist_mat,method="ward"):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)
        
        compute_serial_matrix transforms a distance matrix into 
        a sorted distance matrix according to the order implied 
        by the hierarchical tree (dendrogram)
    '''
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method, preserve_input=True)
    res_order = seriation(res_linkage, N, N + N-2)
    seriated_dist = np.zeros((N,N))
    a,b = np.triu_indices(N,k=1)
    seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]
    
    return seriated_dist, res_order, res_linkage

# Reordering a matrix
#######################################

# Unsupervised feature selection
def rank_features(data_matrix, criterion="variance"):
    sorted_idxs = []
    if criterion == "variance":
        # Highest variance
        variances = np.var(data_matrix, axis=0)
        sorted_idxs = sorted(range(len(variances)), key=lambda i: variances[i], reverse=True)   
    elif criterion == "normalized_std":
        # Highest std/max
        norm_std_devs = np.divide(np.std(data_matrix, axis=0), np.max(data_matrix, axis=0))
        sorted_idxs = sorted(range(len(norm_std_devs)), key=lambda i: norm_std_devs[i], reverse=True)
    else:
        raise ValueError("Select a valid criterion")
    return sorted_idxs
    

def closest_point(point, reference_points):
    # Returns the index of the point in reference_points closest to point (square distance)
    deltas = reference_points - point
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)