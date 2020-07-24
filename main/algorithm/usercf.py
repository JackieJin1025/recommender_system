#! /usr/bin/python3
# coding=utf-8
'''
'''
from collections import defaultdict
from operator import itemgetter
import pandas as pd
import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from main.algorithm.basealgo import BaseAlgo
from main.algorithm.itemcf import ItemCF
from main.util.debug import Timer
from main.util.movielen_reader import load_movielen_data


class UserCF(BaseAlgo):

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
            self.log.warning("%s are not valid" % items[~valid_mask])
            for e in items[~valid_mask]:
                result[e] = np.nan

        items = items[valid_mask]

        upos = self.users.get_loc(user)
        # idx with decending similarities with itself
        full_user_idx = np.argsort(self.user_sim_matrix[upos, :])[::-1][1:]

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
            rating = ratings.dot(sim_wt) / sim_wt.sum()
            result[item] = rating

        df = pd.Series(result)
        return df


if __name__ == '__main__':
    ratings, users, movies = load_movielen_data()
    model = UserCF(min_threshold=0.1, min_nn=5)
    print(model.get_params())
    model.fit(ratings)
    user = 1
    movies = list(movies.item.astype(int))
    df = model.predict_for_user(user, movies)
    print(df.sort_values(ascending=False))