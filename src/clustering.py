from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import dbscan
import pandas as pd
import numpy as np
import abc



#Abstract Clustering class to be implemented for Mapper
class ClusteringTDA(abc.ABC):

    def __init__(self, data, max_clusters, eps, neighbors):
        pass

    @abc.abstractmethod
    def run_clustering(self):
        pass

def find_opt_threshold(hist, bin_edges, limit=3):

    sort_ind = np.lexsort((list(range(len(hist))), hist))

    for i in sort_ind:
        left = i
        right = i
        counter = 0
        while left != 0 and right != len(sort_ind) - 1:
            left -= 1
            right += 1
            if hist[i] < hist[left] and hist[i] < hist[right]:
                counter += 1
            if counter == limit:
                return bin_edges[i]

    return bin_edges[-1]


class SingleLinkageClustering(ClusteringTDA):

    def __init__(self, data, max_clusters, eps, neighbors):
        self.data = data
        self.k = max_clusters
        self.resolution = 0

        self.var_vec = [v if v > 0 else 1. for v in np.var(data, axis=0)]

        self.indices = np.arange(len(data))
        self.ind_to_c = {}
        self.c_to_ind = {}

    def run_clustering(self, lower_popul_bound=2):

        self.resolution, self.hist, self.bin_edges = self.compute_thresh()

        self.tad_algo()

        for c in list(self.c_to_ind.keys()):
            if len(self.c_to_ind[c]) <= lower_popul_bound:
                self.c_to_ind.pop(c, None)

        return self.c_to_ind

    def compute_thresh(self):

        flat_adj_matrix = pdist(self.data, metric='seuclidean', V=self.var_vec)

        hist, bin_edges = np.histogram(flat_adj_matrix, bins=self.k)

        opt_thresh = find_opt_threshold(hist, bin_edges, limit=3)

        return opt_thresh, hist, bin_edges

    def cdistance_norm(self, a, b):
        return cdist(a, b, metric='seuclidean', V=self.var_vec)[0]

    def merge_clusters(self, neighbor_clusters, nodes):

        external_nodes = []

        for c in neighbor_clusters:
            external_nodes.extend(self.c_to_ind[c])
            self.c_to_ind.pop(c, None)

        return list(set(external_nodes) | set(nodes))

    def update_cluster_mmpbership(self, cluster_name):
        return list(zip(self.c_to_ind[cluster_name], [cluster_name] * len(self.c_to_ind[cluster_name])))

    def tad_algo(self):

        cluster_name = 0

        for i in self.indices:

            if i not in self.ind_to_c:
                dists_i = self.cdistance_norm(self.data[i:i + 1], self.data)
                nodes = self.indices[dists_i < self.resolution]

                neighbor_clusters = set([self.ind_to_c[n] for n in nodes if n in self.ind_to_c])

                self.c_to_ind[cluster_name] = self.merge_clusters(neighbor_clusters, nodes)

                clus_mbrship = self.update_cluster_mmpbership(cluster_name)

                self.ind_to_c.update(clus_mbrship)

                cluster_name += 1

class DBSCAN(ClusteringTDA):

    def __init__(self, data, max_clusters, eps, num_neighbors):

        self.data = data
        self.eps = eps
        self.num_neighbors = num_neighbors

        self.ind_to_c = {}
        self.c_to_ind = {}

    def run_clustering(self):

        core_samples, labels = dbscan(self.data, self.eps, self.num_neighbors, metric='euclidean')
        self.ind_to_c = labels

        for i, c_ind in enumerate(self.ind_to_c):
            if c_ind == -1:
                pass
            else:
                try:
                    keys = list(self.c_to_ind.keys())
                    index = keys.index(c_ind)
                    self.c_to_ind[c_ind].append(i)
                except ValueError:
                    self.c_to_ind[c_ind] = [i]

        return self.c_to_ind

class NNC(ClusteringTDA):

    def __init__(self, data, max_clusters, eps, neighbors):
        self.data = data
        self.k = max_clusters

        self.var_vec = [v if v > 0 else 1. for v in np.var(data, axis=0)]

        self.indices = np.arange(len(data))
        self.ind_to_c = {}
        self.c_to_ind = {}

    def run_clustering(self, lower_popul_bound=0):

        self.nnc_algo()

        for c in list(self.c_to_ind.keys()):
            if len(self.c_to_ind[c]) <= lower_popul_bound:
                self.c_to_ind.pop(c, None)

        return self.c_to_ind

    def cdistance_norm(self, a, b):
        return cdist(a, b, metric='seuclidean', V=self.var_vec)[0]

    def merge_clusters(self, neighbor_clusters, nodes):

        external_nodes = []

        for c in neighbor_clusters:
            external_nodes.extend(self.c_to_ind[c])
            self.c_to_ind.pop(c, None)

        return list(set(external_nodes) | set(nodes))

    def update_cluster_mmpbership(self, cluster_name):
        return list(zip(self.c_to_ind[cluster_name], [cluster_name] * len(self.c_to_ind[cluster_name])))

    def nnc_algo(self):

        cluster_name = 0

        for i in self.indices:

            if i not in self.ind_to_c:
                dists_i = self.cdistance_norm(self.data[i:i + 1], self.data)

                nodes = self.indices[np.argsort(dists_i)[:self.k]][1:]

                neighbor_clusters = set([self.ind_to_c[n] for n in nodes if n in self.ind_to_c])

                self.c_to_ind[cluster_name] = self.merge_clusters(neighbor_clusters, nodes)

                clus_mbrship = self.update_cluster_mmpbership(cluster_name)

                self.ind_to_c.update(clus_mbrship)

                cluster_name += 1