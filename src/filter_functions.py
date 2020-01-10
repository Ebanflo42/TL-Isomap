from scipy.spatial.distance import cdist
import pandas as pd
import numpy as np

"""
Filter functions take data, plus some additional paramaters,
and return a function which takes an index from the data and outputs a real number;
they are meant to represent the discrete version of a Morse function.
"""

def eccentricity_p(data, p):
    """
    :param data: point cloud in Euclidean space
    :type data: 2D numpy array
    :param p: exponent for L^p distance
    :type p: floating point number greater than or equal to 1.0
    :return: a function providing average L^p distance to every other point
    """

    covariance = [v if v > 0 else 1. for v in np.var(data, axis=0)]

    def _fin_ecc(x, data):
        num = np.power(cdist([x], data, metric='seuclidean', V=covariance)[0], p)
        result = np.sum(num) / len(data)
        return np.power(result, 1. / p)

    def _inf_ecc(x, data):
        return np.max(cdist([x], data, metric='seuclidean', V=covariance)[0])

    if p == 'inf':
        return (lambda i: _inf_ecc(data[i], data))
    else:
        return (lambda i: _fin_ecc(data[i], data))


def axis_proj(data, coordinate):
    """
    :param data: point cloud in Euclidean space
    :type data: 2D numpy array
    :param coordinate: which coordinate to project onto
    :type coordinate: integer greater than or equal to 0 and less than the dimension of the data
    :return: a function providing the projection of each point onto the specified coordinate
    """
    return (lambda i: data[i][coordinate])

def arc_len(data, traj_len):
    """
    :param data: set of trajectories in Euclidean space
    :type data: 2D numpy array
    :param traj_len: length of each trajectory (each trajectory must have the same length)
    :type traj_len: positive integer which divides the length of the data set
    :return: a function providing the arc length of a given point along its trajectory
    """

    if len(data)%traj_len != 0:
        raise ValueError('The length of the data set must be a multiple of the trajectory length (all trajectories must have the same length).')

    else:

        num_trajs = len(data)//traj_len
        arc_len_arr = num_trajs*[traj_len*[None]]

        for i in range(num_trajs - 1):
            arc_len_arr[i][0] = 0
            for j in range(traj_len - 1):
                arc_len_arr[i][j + 1] = arc_len_arr[i][j] + cdist([data[traj_len*i + j]], [data[traj_len*i + j + 1]], metric='euclidean')[0][0] #there's got to be a better way...

        return (lambda i: arc_len_arr[i//traj_len][i%traj_len])
