from collections import defaultdict
from operator import itemgetter
import pickle

import bottleneck
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from main.algorithm.basic import Predictor
from main.algorithm.bias import Bias
from main.utils.data import get_data, load_movielen_data
import os
import pandas as pd

from main.utils.debug import LogUtil, Timer
from main.utils.functions import _demean, _norm, _nn_score


class ItemCF(Predictor):

    def __init__(self,
                 min_nn=1, 
                 max_nn=50,
                 min_threshold=None,
                 bias = None,
                 *args, **kwargs):
        self.bias = bias
        self.min_nn = min_nn 
        self.max_nn = max_nn 
        self.min_threshold = min_threshold

        self.item_sim_matrix = None
        super(ItemCF, self).__init__(*args, **kwargs)

    def fit(self, data):
        if self.bias is not None:
            self.bias.fit(data)
        super(ItemCF, self).fit(data)

    def _train(self):
        self.item_sim_matrix = self.item_similarity()

    def _save(self):
        # cache trained parameter
        with open(self.filename, 'wb') as outfile:
            pickle.dump(self.item_sim_matrix, outfile)
        self.log.info("save item_sim_matrix to %s", self.filename)

    def _load(self):
        with open(self.filename, 'rb') as infile:
            self.item_sim_matrix = pickle.load(infile)
        self.log.info("loaded item_sim_matrix from %s", self.filename)

    def item_similarity(self):
        # user by item
        rmat = self.rmat.tocsc()
        #normalize by item
        _ = _demean(rmat)
        _ = _norm(rmat)
        item_sim = rmat.T.dot(rmat).toarray()
        return item_sim

    def predict_for_user(self, user, items=None, ratings=None):
        if items is not None:
            items = np.array(items)
        else:
            items = self.items.values
        valid_mask = self.items.get_indexer(items) >= 0

        min_threshold = self.min_threshold
        min_nn = self.min_nn
        max_nn = self.max_nn
        item_sim = self.item_sim_matrix
        result = dict()
        # convert rmat to array
        rmat_array = self.rmat.toarray()

        if np.sum(~valid_mask) > 0:
            self.log.debug("user %s: %s are not valid", user, items[~valid_mask])
            for e in items[~valid_mask]:
                result[e] = np.nan

        items = items[valid_mask]
        upos = self.users.get_loc(user)
        item_bias = None
        if self.bias is not None:
            item_bias = self.bias.get_item_bias()

        for item in items:
            ipos = self.items.get_loc(item)

            # idx with descending similarities with itself
            clock = Timer()
            item_idx = np.argsort(item_sim[ipos, :])[::-1]
            #item_idx = bottleneck.argpartition(item_sim[ipos, :], 3000)
            e0 = clock.restart()

            # sim need to meet min_threshold
            if min_threshold is not None:
                item_idx = item_idx[item_sim[ipos, item_idx] > min_threshold]
            if len(item_idx) < min_nn:
                self.log.debug("item %s does not have enough neighbors (%s < %s)", item, len(item_idx), min_nn)
                result[item] = np.nan
                continue

            # narrow down to items were rated
            item_idx = item_idx[rmat_array[upos, item_idx] != 0]
            item_idx = item_idx[:max_nn]
            item_scores = rmat_array[upos, :]
            e1 = clock.restart()
            rating = _nn_score(item_scores, item_sim, ipos, item_idx, item_bias)
            e2 = clock.restart()
            # print(e0, e1, e2)
            result[item] = rating

        df = pd.Series(result)
        return df


if __name__ == '__main__':
    LogUtil.configLog()
    ratings, users, movies = load_movielen_data()
    bias = Bias()
    itemcf = ItemCF(min_threshold=0.1, min_nn=5, bias=bias)
    print(itemcf.get_params())
    itemcf.fit(ratings)
    user = 1
    movies = list(movies.item.astype(int))
    clock = Timer()
    for i in range(10):
        df = itemcf.predict_for_user(user, movies)
        print(clock.restart())

    print(df.describe())