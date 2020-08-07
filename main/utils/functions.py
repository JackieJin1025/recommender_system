import numpy as np
from numba import njit


def _demean(sparse):
    """
    demean sparse matrix along row if sparse is csr or along column if sparse is csc
    :param sparse: csr or csc
    :return:
    """
    n = sparse.shape[1]
    if sparse.getformat() == 'csr':
        n = sparse.shape[0]

    indptr = sparse.indptr
    data = sparse.data
    indices = sparse.indices
    means = np.zeros(n)
    for i in range(n):
        sp, ep = indptr[i], indptr[i +1]
        if sp == ep:
            continue  # empty row
        vs = data[sp:ep]
        m = np.mean(vs)
        means[i] = m
        sparse.data[sp:ep] -= m
    return means


def _norm(sparse):
    """
    normalize sparse matrix along row if sparse is csr or along column if sparse is csc
    :param sparse: csr or csc
    :return:
    """
    n = sparse.shape[1]
    if sparse.getformat() == 'csr':
        n = sparse.shape[0]

    indptr = sparse.indptr
    data = sparse.data
    indices = sparse.indices
    norms = np.ones(n)
    for i in range(n):
        sp, ep = indptr[i], indptr[i +1]
        if sp == ep:
            continue # empty row
        vs = data[sp:ep]
        m = np.linalg.norm(vs)
        if m == 0:
            continue
        norms[i] = m
        sparse.data[sp:ep] /= m
    return norms


@njit
def _nn_score(scores, sim_matrix, idx, nb_idx, bias=None):
    """
    :param scores: 1D array with scores (scores from all neighbors)
    :param sim_matrix: 2D array with similarity matrix
    :param idx: index of user/item which to be scored
    :param nb_idx: neighbor index
    :param bias: 1D array bias
    :return:
    """

    n = len(nb_idx)
    scores = scores[nb_idx]
    neighbor_bias, own_bias = np.zeros(n), 0.0
    if bias is not None:
        neighbor_bias = bias[nb_idx]
        own_bias = bias[idx]

    scores = scores - neighbor_bias
    sim_wt = sim_matrix[idx][nb_idx]
    if sim_wt.sum() <= 0:
        return np.nan
    result = own_bias + scores.dot(sim_wt) / sim_wt.sum()


    # for i in range(len(scores)):
    #     scores[i] = scores[i] - neighbor_bias[i]
    # tot = 0.0
    # tot_wt = 0.0
    # for i in range(n):
    #     tot += scores[i] * sim_wt[i]
    #     tot_wt += sim_wt[i]
    # if tot_wt <= 0:
    #     return np.nan
    # result = own_bias + tot / tot_wt

    return result