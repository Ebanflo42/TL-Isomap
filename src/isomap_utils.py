import numpy as np
import numpy.linalg as la
import networkx as nx

def build_nbrhd_graph(arr_2d, d):
    """
    :param arr_2d: point cloud in Euclidean space
    :type arr_2d: 2D numpy array
    :param d: maximum distance for which an edge will be constructed
    :type d: floating point
    :return: networkx graph of neighbors within d distance; labeled with squared distances
    """

    result = nx.Graph()
    num_points = len(arr_2d)

    for i in range(0, num_points - 1):
        for j in range(i + 1, num_points - 1):
            dist = la.norm(arr_2d[i] - arr_2d[j])
            if dist < d:
                dist *= dist
                result.add_edge(i, j, weight = dist)

    return result

def floyd_warshall(graph, lm_verts):
    """
    :param graph: weighted graph
    :type graph: networkx.Graph with weight attribute on each edge
    :param lm_verts: landmarked vertices of the graph
    :type lm_verts: list of indices into the nodes of the graph
    :param v: number of vertices
    :type v: int
    :return: shortest distance from any point to any landmark point as 2D numpy array
    """

    num_verts = len(list(graph.nodes))

    dist = np.array([np.array([float("inf") for _ in range(num_verts)]) for _ in range(num_verts)])

    for n in list(graph.nodes):
        n_edges = nx.edges(graph, n)
        for (i, j) in n_edges:
            dist[i][j] = graph[i][j]['weight']
            dist[j][i] = graph[i][j]['weight']

    for k in lm_verts:
        for i in range(num_verts):
            for j in range(num_verts):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    dist[j][i] = dist[i][k] + dist[k][j]

    return dist

def mds(k, sqr_dist_mat):
    """
    :param k: desired output dimension
    :type k: int
    :param sqr_dist_mat: matrix of squared distances between nodes of a weighted graph
    :type sqr_dist_mat: 2D numpy array
    :return: k-dimensional embedding of the original data as a 2D numpy array
    """

    if len(sqr_dist_mat) != len(sqr_dist_mat[0]):
        ValueError('isomap_utils.mds: distance matrix not square')

    n = len(sqr_dist_mat)
    centerer = (1.0/n)*np.ones((n, n))
    centered_mat = -0.5*np.matmul(centerer, np.matmul(sqr_dist_mat, centerer))
    values, vectors = la.eigh(centered_mat)

    #k may have to be reduced if the intrinsic dimension of the data is lower
    m = 0
    for i in range(n):
        if values[i] <= 0:
            m += 1
    if m < k: k = m

    important_values = values[n - k : n]
    important_vectors = vectors[n - k : n]
    embedding_matrix = np.diag(np.sqrt(important_values)) @ important_vectors
    return important_values, embedding_matrix

def lmds(k, sqr_dist_mat):
    """
    :param k: desired output dimension
    :type k: int
    :param sqr_dist_mat: matrix of squared distances between landmark nodes and all other nodes
    :type sqr_dist_mat: 2D numpy array
    :return: embedding of the original data in k dimensions as 2D numpy array
    """

    num_landmarks = len(sqr_dist_mat)
    num_points = len(sqr_dist_mat[0])
    sub_mat = sqr_dist_mat[:, num_points - num_landmarks : num_points]
    evalues, landmark_embedding = mds(k, sub_mat)

    pseudo_embedding = np.diag(np.reciprocal(evalues)) @ landmark_embedding
    sum_sq_dist = np.sum(sqr_dist_mat[:, 0 : num_landmarks - 1])
    mean_sq_dist = np.transpose(np.multiply(1.0/num_landmarks, sum_sq_dist))
    rest_of_dist = sqr_dist_mat[:, num_landmarks : num_points]

    for i in range(0, len(rest_of_dist - 1)):
        helper = rest_of_dist[:, i]
        rest_of_dist[:, i] = helper - mean_sq_dist

    rest_of_embedding = np.transpose(pseudo_embedding @ rest_of_dist)

    return rest_of_embedding
