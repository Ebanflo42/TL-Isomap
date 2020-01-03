import numpy as np
import numpy.linalg as la

def mds(k, sqr_dist_mat):

    """
    Classical multidimensional scaling.
    First argument is the desired dimension and second is the matrix of squared distances.
    """

    n = len(sqr_dist_mat)
    centerer = np.multiply(1.0/n, np.ones(n, n))
    centered_mat = np.multiply(-0.5, np.mutiply(centerer, np.multiply(sqr_dist_mat, centerer)))
    values, vectors = la.eigh(centered_mat)

    for i in range(0, n - 1):
        if values[i] <= 0:
            k -= 1

    important_values = values[n - k - 1 : n - 1]
    important_vectors = vectors[n - k - 1 : n - 1]
    embedding_matrix = np.multiply(important_vectors, np.diag(np.sqrt(important_values)))
    return important_values, embedding_matrix

def lmds(k, sqr_dist_mat):

    """
    Landmark multidimensional scaling.
    First argument is desired dimension and second argument is squared distance matrix;
    each row corresponds to a landmark point and each entry in the row is the distance
    to a different data point, it is assumed that the leftmost square submatrix is the
    symmetric matrix of distances between landmark points.
    Here we discard the landmark points because, for topological L-isomap, the landmarks
    come from outside the data.
    """

    num_landmarks = len(sqr_dist_mat)
    num_points = len(sqr_dist_mat[0])
    sub_mat = sqr_dist_mat[:, 0 : num_landmarks - 1]
    evalues, landmark_embedding = mds(k, sub_mat)

    k = len(landmark_embedding[:, 0])
    pseudo_embedding = np.multiply(landmark_embedding, np.diag(np.reciprocal(evalues)))
    sum_sq_dist = np.sum(sqr_dist_mat[:, 0 : num_landmarks - 1])
    mean_sq_dist = np.transpose(np.multiply(1.0/num_landmarks, sum_sq_dist))
    rest_of_dist = sqr_dist_mat[:, num_landmarks : num_points - 1]

    for i in range(0, len(rest_of_dist - 1)):
        helper = rest_of_dist[:, i]
        rest_of_dist[:, i] = np.substract(helper, mean_sq_dist)

    rest_of_embedding = np.multiply(-0.5, np.multiply(pseudo_embedding, rest_of_dist))

    return rest_of_embedding
