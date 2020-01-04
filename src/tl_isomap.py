import networkx as nx
import pandas as pd
import numpy as np
import types

import clustering as cl
import isomap_utils as iu

class TLIsomap:

    #note that max_clusters does not apply to DBSCAN, and eps and num_neighbors only apply to DBSCAN
    def __init__(self,
                 data,
                 filter_function,
                 Clustering,
                 overlap = 50,
                 num_bins = 20,
                 max_clusters = 10,
                 eps = 0.5,
                 num_neighbors = 10):

        self.overlap = overlap
        self.num_bins = num_bins

        self.max_clusters = max_clusters
        self.eps = eps
        self.num_neighbors = num_neighbors

        self.data = data
        self.indices = np.arange(len(data))

        self.filter_function = filter_function
        self.cluster_class = Clustering

        self.filter_values = None
        self.clusters = None
        self.centroids = None
        self.graph = None

        self._check_implem()

    def _check_implem(self):

        if not isinstance(self.filter_function, types.LambdaType):
            raise TypeError('`filter_function` must be callable.')

        if not issubclass(self.cluster_class, cl.ClusteringTDA):
            raise TypeError('`cluster_class` must be an instance of clustering.ClusteringTDA.')

    def _apply_filter_function(self):

        fm = []
        for i in self.indices:
            fm.append(self.filter_function(i))

        self.filter_values = pd.Series(fm, index=self.indices).sort_values()

    def _bin_data(self):
        """
         Bin filter function array into N bins with percent overlap given by self.overlap
         Return filter function bin membership and the bins themselves
        """

        finish = self.filter_values.iloc[-1]
        start = self.filter_values.iloc[0]

        bin_len = (finish-start)/self.num_bins
        bin_over = self.overlap*bin_len
        bins = [(start + (bin_len-bin_over)*i, start + bin_len*(i+1)) for i in range(self.num_bins)]

        binned_dict = {}
        for interval in bins:
            is_member = self.filter_values.apply(lambda x: x >= interval[0] and x <= interval[1])
            binned_dict[interval] = self.filter_values[is_member]

        return binned_dict, bins

    def _apply_clustering(self):
        binned_dict, bins = self._bin_data()

        self.clusters = {}
        counter = 0

        for i, interval in enumerate(bins):

            keys = list(binned_dict[interval].index)

            local_to_global = dict(zip(list(range(len(self.data))), keys))

            cluster_obj = self.cluster_class(self.data[keys],
                                             self.max_clusters,
                                             self.eps,
                                             self.num_neighbors)

            cluster_to_ind = cluster_obj.run_clustering()

            global_cluster_names = {}
            for cluster in cluster_to_ind.keys():
                global_cluster_names[counter] = [local_to_global[ind] for ind in cluster_to_ind[cluster]]
                counter += 1

            self.clusters[i] = global_cluster_names

    def _build_graph(self):

        self.graph = nx.Graph()

        for k in range(len(self.clusters) - 1):
            for c in self.clusters[k]:
                self.graph.add_node(c)

        for k in range(len(self.clusters) - 1):
            for c1 in self.clusters[k]:
                for c2 in self.clusters[k + 1]:
                    if set(self.clusters[k][c1]).intersection(self.clusters[k + 1][c2]):
                        self.graph.add_edge(c1, c2)

        self.graph = self.graph

    def _get_centroids(self):

        c_to_centroid = {}
        for _, clusters in self.clusters.items():
            for node, indices in clusters.items():
                c_to_centroid[node] = np.mean(self.data[indices], axis=0)

        self.centroids = c_to_centroid

    def run_mapper(self):

        self._apply_filter_function()
        self._apply_clustering()
        self._build_graph()

    def embed_k_dims(self, k):

        if self.graph == None:
            self.run_mapper()

        self._get_centroids()

        augmented_data = np.append(self.data, self.centroids)

        nbrhd_graph = nx.Graph()
