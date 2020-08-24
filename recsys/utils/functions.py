import numpy as np
from numba import njit
import pandas as pd

from recsys.utils.data import load_movielen_data, sparse_ratings


def _demean(sparse, damping=0):
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
        m = np.sum(vs) / (damping + len(vs))
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


def _get_xs(sparse, ipos):
    """
    get a column/row from csc/csr
    :param sparse: csc or csr. if csc, get ipos-th col. Otherwise, get ipos-th row
    :param ipos:
    :return:
    """
    # number of rows
    n = sparse.shape[0]
    if sparse.getformat() == 'csr':
        # number of cols
        n = sparse.shape[1]

    result = np.zeros(n)
    sp, ep = sparse.indptr[ipos], sparse.indptr[ipos+1]
    data = sparse.data[sp:ep]
    idx = sparse.indices[sp:ep]
    result[idx] = data
    return result





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


def scores_to_series(scores, item_series, items):
    """

    :param scores: 1d array a list of scores
    :param item_series: item index map, it shoud
    :param items: a list/ array of item
    :return: a series with items as index, and scores as value
    """
    n1, n2 = len(scores), len(item_series)
    if n1 != n2:
        raise AssertionError("len(scores) %d != len(item_series) %d" % (n1, n2))

    item_idx = item_series.get_indexer(items)
    invalid_items = items[item_idx == -1]
    df = pd.Series(data=np.NAN, index=invalid_items)
    item_idx = item_idx[item_idx >= 0]
    items = item_series[item_idx]
    scores = scores[item_idx]
    df = df.append(pd.Series(data=scores, index=items))
    return df


def cosine(sparse_csr, min_sprt=5):
    """not ready to use yet """
    nrow, ncol = sparse_csr.shape
    sparse_coo = sparse_csr.tocoo(copy=True)
    rating = pd.DataFrame([sparse_coo.row.astype(int), sparse_coo.col.astype(int), sparse_coo.data])
    n_x = len(sparse_coo.data)

    prods = np.zeros((nrow, nrow), np.double)
    freq = np.zeros((nrow, nrow), np.int)
    sqi = np.zeros((nrow, nrow), np.double)
    sqj = np.zeros((nrow, nrow), np.double)
    sim = np.zeros((nrow, nrow), np.double)

    for upos_i, ipos_i, score_i in rating:
        for upos_j, ipos_j, score_j in rating:
                prods[upos_i, upos_j] += score_i * score_j
                freq[upos_i, upos_j] += 1
                sqi[upos_i, upos_j] += score_i**2
                sqj[upos_i, upos_j] += score_j**2

    for xi in range(n_x):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n_x):
            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                denum = np.sqrt(sqi[xi, xj] * sqj[xi, xj])
                sim[xi, xj] = prods[xi, xj] / denum

            sim[xj, xi] = sim[xi, xj]

    return sim


if __name__ == '__main__':
    ratings, users, movies = load_movielen_data()
    rmat, _, _, = sparse_ratings(ratings)
    print("xxx")
    print(cosine(rmat))
