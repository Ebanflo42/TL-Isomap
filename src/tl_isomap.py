import networkx as nx
import pandas as pd
import numpy as np
import numpy.linalg as la
import types
import warnings

import clustering as cl
import isomap_utils as iso

"""
This module implements Topological L-isomap and (by necessity) the mapper algorithm.

The TLIsomap constructor has the following fields
data: the data to be analyzed as a 2D numpy array
filter_function: function taking an index into the data set and returning a real number
Clustering: A clustering class, see clustering.py
overlap: percent overlap between the intervals used for mapper
num_bins: number of intervals to cut the image of the filter function into
max_slc_clusters: maximum number of clusters that SLC may identify
dbscan_eps: how close must two points be for DBSCAN to consider them neighbors
num_neighbors: how many neighbors must a point have for DBSCAN to consider it a core point
isomap_eps: how close must two points be for them to have an edge in the neighborhood graph

Once a TLIsomap object is initialized, run_mapper() will exectute the mapper algorithm.
The graph attribute contains the resulting Reeb graph as a networkx graph,
and the centroids attribute contains the centroids of each cluster as
a dictionary whose keys are the nodes of the graph.

embed_k_dims(k) will return the data embedded the data into k dimensions using Topological L-isomap.
None of the helper variables for this algorithm are stored as fields in the class.
If run_mapper() has not been executed, this function will execute it first,
but not re-execute it unnecessarily.
"""

class TLIsomap:

    def __init__(self,
                 data,
                 filter_function,
                 Clustering,
                 overlap=50,
                 num_bins=20,
                 max_slc_clusters=10,
                 dbscan_eps=0.5,
                 num_neighbors=10,
                 isomap_eps=0.5):

        self.overlap = overlap
        self.num_bins = num_bins

        self.max_slc_clusters = max_slc_clusters
        self.dbscan_eps = dbscan_eps
        self.num_neighbors = num_neighbors
        self.isomap_eps = isomap_eps

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
                                             self.max_slc_clusters,
                                             self.dbscan_eps,
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
                c_to_centroid[node] = np.mean(self.data[indices], axis = 0)

        self.centroids = c_to_centroid

    def run_mapper(self):

        self._apply_filter_function()
        self._apply_clustering()
        self._build_graph()
        self._get_centroids()

    def embed_k_dims(self, k):

        if self.graph == None:
            self.run_mapper()

        num_datum = len(self.data)
        num_landmarks = len(self.graph.nodes)
        if num_landmarks == 0:
            ValueError('tl_isomap.py: No landmarks!')
        total = num_datum + num_landmarks
        augmented_data = self.data
        for node in list(self.graph.nodes):
            augmented_data = np.append(augmented_data, self.centroids[node])

        nbhrd_graph = iso.build_nbrhd_graph(augmented_data, self.isomap_eps)
        for i, j in list(self.graph.edges):
            dist = la.norm(self.centroids[i] - self.centroids[j])
            nbhrd_graph.add_edge(num_datum + i, num_datum + j, weight = dist*dist)

        if not nx.is_connected(nbhrd_graph):
            warnings.warn('The neighborhood grpha is disconnected; TL-Isomap will embed only one of the components.', RuntimeWarning)

        dist_mat = iso.floyd_warshall(nbhrd_graph, range(num_datum, total))
        sub_mat = dist_mat[total - num_landmarks - 1 : total - 1]

        embedding = iso.lmds(k, sub_mat)

        return embedding