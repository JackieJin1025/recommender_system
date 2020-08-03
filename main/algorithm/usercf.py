from collections import defaultdict
from operator import itemgetter
import pandas as pd
import os
import pickle
import numpy as np
from numba import njit
from sklearn.metrics.pairwise import cosine_similarity

from main.algorithm.basic import Predictor
from main.algorithm.itemcf import ItemCF
from main.util.debug import Timer
from main.util.data import load_movielen_data
from heapq import heapify, heappop, heappush

class UserCF(Predictor):

    def __init__(self, discount_popularity=False,
                 min_nn=1,
                 max_nn=50,
                 min_threshold=None,
                 *args, **kwargs):
        self.discount_popularity = discount_popularity
        self.user_sim_matrix = None
        self.min_nn = min_nn
        self.max_nn = max_nn
        self.min_threshold = min_threshold

        super(UserCF, self).__init__(*args, **kwargs)

    def _train(self):
        """
            train model
        """
        self.user_sim_matrix = self.user_similarity()

    def _save(self):
        c = Timer()
        # cache trained parameter
        with open(self.filename, 'wb') as outfile:
            pickle.dump(self.user_sim_matrix, outfile)
        e = c.restart()

        self.log.info("saving user_sim_matrix to %s takes %.3f", self.filename, e)

    def _load(self):
        c = Timer()
        with open(self.filename, 'rb') as infile:
            self.user_sim_matrix = pickle.load(infile)
        e = c.restart()
        self.log.info("loading user_sim_matrix from %s takes %.3f", self.filename, e)

    def user_similarity(self):
        """
        :return: user by user similarity matrix
        """
        # user by item
        rmat = self.rmat
        #normalize by user norm
        rmat_array = rmat.toarray()
        user_norm = np.linalg.norm(rmat_array, axis=1)
        nmat = rmat_array / user_norm.reshape(-1, 1)
        user_sim = nmat.dot(nmat.T)
        user_sim_diag = np.diag(user_sim)

        epsilon = 1e-6
        b1 = user_sim_diag > 1 + epsilon
        b2 = user_sim_diag < 1 - epsilon
        idx = np.argwhere(b1 | b2).flatten()
        if len(idx) > 0:
            self.log.warning("diagonals of similarity matrix (%s) = %s, Is this expected? ", idx, user_sim_diag[idx])

        return user_sim

    def predict_for_user(self, user, items=None, ratings=None):
        """

        :param user: user id
        :param items: a list of item ids
        :param ratings:
        :return:
        """

        min_threshold = self.min_threshold
        min_nn = self.min_nn
        max_nn = self.max_nn

        result = dict()
        # convert rmat to array
        rmat_array = self.rmat.toarray()
        if items is not None:
            items = np.array(items)
        else:
            items = self.items.values

        valid_mask = self.items.get_indexer(items) >= 0
        if np.sum(~valid_mask) > 0:
            # self.log.warning("%s are not valid" % items[~valid_mask])
            for e in items[~valid_mask]:
                result[e] = np.nan

        items = items[valid_mask]

        upos = self.users.get_loc(user)
        # idx with decending similarities with itself
        full_user_idx = np.argsort(self.user_sim_matrix[upos, :])[::-1]

        if False:
            min_sim = self.user_sim_matrix[upos, full_user_idx].min()
            max_sim = self.user_sim_matrix[upos, full_user_idx].max()
            self.log.info("max similarity and min similarity are %.3f and %.3f", max_sim, min_sim)

        # sim need to meet min_threshold
        if min_threshold is not None:
            full_user_idx = full_user_idx[self.user_sim_matrix[upos, full_user_idx] > min_threshold]

        for item in items:
            ipos = self.items.get_loc(item)

            # narrow down to users who rated the item
            user_idx = full_user_idx[rmat_array[full_user_idx, ipos] != 0]
            user_idx = user_idx[:max_nn]
            if len(user_idx) < min_nn:
                self.log.debug("user %s does not have enough neighbors (%s < %s)", user, len(user_idx), min_nn)
                result[item] = np.nan
            ratings= rmat_array[user_idx, ipos]
            sim_wt = self.user_sim_matrix[upos, user_idx]
            if sim_wt.sum() <= 0:
                result[item] = np.nan
                continue
            rating = ratings.dot(sim_wt) / sim_wt.sum()
            result[item] = rating

        df = pd.Series(result)
        return df

    def predict_for_user_numba(self, user, items=None, ratings=None):
        """
        Doesn't not seem to improve the performance !!!
        who can make it better ?
        :param user: user id
        :param items: a list of item ids
        :param ratings:
        :return:
        """
        min_threshold = self.min_threshold
        min_nn = self.min_nn
        max_nn = self.max_nn
        rmat_array = self.rmat.toarray()
        if items is not None:
            items = np.array(items)
        else:
            items = self.items.values

        items_idx = self.items.get_indexer(items)
        upos = self.users.get_loc(user)
        usims = self.user_sim_matrix[upos, :]

        result = _score(rmat_array, usims, items_idx, min_threshold, min_nn, max_nn)
        return pd.Series(result, index=items)

@njit
def _score(rmat_array, usims, items_idx, min_threshold, min_nn, max_nn):
    """
    :param rmat_array: rating matrix, m x n where m is the number of users, n is the number of items
    :param usims: 1d array of user similarity 1 x m
    :param items_idx: 1d array of item indexes
    :param min_threshold: minimal similarity to be considered as neighbor
    :param min_nn:  int min number of neighbors to get a score
    :param max_nn:  int max number of neighbors to get a score
    :return:
    """

    result = np.full(len(items_idx), np.nan, dtype=np.float32)
    u_mask = np.full(len(usims), 1, dtype=np.bool_)
    rmat_t = rmat_array.T
    if min_threshold is not None:
        u_mask = usims >= min_threshold

    for i, item_idx in enumerate(items_idx):
        if item_idx < 0:
            continue
        ratings = rmat_t[item_idx, :]
        # only for users who have ratings
        mask = ratings != 0
        if min_threshold is not None:
            mask = np.bitwise_and(mask.astype(np.bool_), u_mask.astype(np.bool_))

        cnt = 0
        for j in range(len(mask)):
            if mask[j]:
                cnt += 1
        if cnt < min_nn:
            continue

        # ratings = ratings[mask]
        # valid_usims = usims[mask]
        #
        # heap = [(-valid_usims[0], 0)]
        # for j, sim in enumerate(valid_usims, 1):
        #     heappush(heap, (-sim, j))
        #
        # numerator = 0.0
        # denominator = 0.0
        # while len(heap) > 0 and max_nn > 0:
        #     sim, j = heappop(heap)
        #     sim = -sim
        #     numerator += ratings[j] * sim
        #     denominator += sim
        #     max_nn -= 1
        # if denominator <= 0:
        #     continue
        # score = numerator / denominator
        # result[i] = score

        ratings = ratings[mask]
        valid_usims = usims[mask]
        sorted_uidx = np.argsort(valid_usims)[::-1]
        sorted_uidx = sorted_uidx[:max_nn]
        ratings = ratings[sorted_uidx]
        valid_usims = valid_usims[sorted_uidx]
        if valid_usims.sum() <= 0:
            continue
        score = ratings.dot(valid_usims) / valid_usims.sum()
        result[i] = score

    return result


if __name__ == '__main__':
    ratings, users, movies = load_movielen_data()
    model = UserCF(min_threshold=0.1, min_nn=5)
    print(model.get_params())
    model.fit(ratings)
    user = 1
    movies = list(movies.item.astype(int))
    # movies = [2]
    clock = Timer()
    for _ in range(5):
        df = model.predict_for_user(user, movies)
        print(clock.restart())

    print("="*60)
    for _ in range(5):
        df2 = model.predict_for_user_numba(user, movies)
        print(clock.restart())

    print(df.sort_values(ascending=False).head(5))
    print(df2.sort_values(ascending=False).head(5))
    print(df.describe())
    print(df2.describe())

    df_s = pd.DataFrame({'old': df,
                         'new': df2})

    print(df_s[df_s.old != df_s.new])