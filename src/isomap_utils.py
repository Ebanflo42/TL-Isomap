import numpy as np
import numpy.linalg as la
import networkx as nx

def nbrhd_graph(arr_2d, d):
    """
    :param arr_2d: point cloud in Euclidean space
    :type arr_2d: 2D numpy array
    :param d: maximum distance for which an edge will be constructed
    :type d: floating point
    :return: networkx graph of neighbors within d distance; labeled with squared distances
    """

    result = nx.graph()
    num_points = len(arr_2d)

    for i in range(0, num_points - 1):
        for j in range(i + 1, num_points - 1):
            dist = la.norm(arr_2d[i] - arr_2d[j])
            if dist < d:
                dist *= dist
                result.add_edge(i, j, weight=dist)

    return result

def floyd_warshall(graph, v):
    """
    :param graph: weighted graph
    :type graph: networkx.Graph with weight attribute on each edge
    :param v: number of vertices
    :type v: int
    :return: shortest distance between all vertex pairs as 2D numpy array
    """

    dist = np.array([np.array([float("inf") for _ in range(v)]) for _ in range(v)])

    for n in list(graph.nodes):
        n_edges = nx.edges(graph, n)
        for (i, j) in n_edges.data('weight'):
            dist[i][j] = graph[i][j].weight

    # check vertex k against all other vertices (i, j)
    for k in range(v):
        # looping through rows of graph array
        for i in range(v):
            # looping through columns of graph array
            for j in range(v):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist

def mds(k, sqr_dist_mat):

    """
    :param k: desired output dimension
    :type k: int
    :param sqr_dist_mat: matrix of squared distances between nodes of a weighted graph
    :type sqr_dist_mat: 2D numpy array
    :return: k-dimensional embedding of the original data as a 2D numpy array
    """

    n = len(sqr_dist_mat)
    centerer = np.multiply(1.0/n, np.ones(n, n))
    centered_mat = np.multiply(-0.5, np.mutiply(centerer, np.multiply(sqr_dist_mat, centerer)))
    values, vectors = la.eigh(centered_mat)

    #k may have to be reduced if the intrinsic dimension of the data is lower
    for i in range(0, n - 1):
        if values[i] <= 0:
            k -= 1

    important_values = values[n - k - 1 : n - 1]
    important_vectors = vectors[n - k - 1 : n - 1]
    embedding_matrix = np.multiply(important_vectors, np.diag(np.sqrt(important_values)))
    return important_values, embedding_matrix

def lmds(k, sqr_dist_mat):

    """
    :param k: desired output dimension
    :type k: int
    :param sqr_dist_mat: matrix of squared distances between landmark nodes and all other nodes
    :type sqr_dist_mat: 2D numpy array
    """

    num_landmarks = len(sqr_dist_mat)
    num_points = len(sqr_dist_mat[0])
    sub_mat = sqr_dist_mat[:, 0 : num_landmarks - 1]
    evalues, landmark_embedding = mds(k, sub_mat)

    pseudo_embedding = np.multiply(landmark_embedding, np.diag(np.reciprocal(evalues)))
    sum_sq_dist = np.sum(sqr_dist_mat[:, 0 : num_landmarks - 1])
    mean_sq_dist = np.transpose(np.multiply(1.0/num_landmarks, sum_sq_dist))
    rest_of_dist = sqr_dist_mat[:, num_landmarks : num_points - 1]

    for i in range(0, len(rest_of_dist - 1)):
        helper = rest_of_dist[:, i]
        rest_of_dist[:, i] = np.substract(helper, mean_sq_dist)

    rest_of_embedding = np.multiply(-0.5, np.multiply(pseudo_embedding, rest_of_dist))

    return rest_of_embedding
