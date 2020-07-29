from collections import defaultdict
from operator import itemgetter
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from main.algorithm.basealgo import BaseAlgo
from main.util.data import get_data
import os
import pandas as pd

from main.util.movielen_reader import load_movielen_data


class ItemCF(BaseAlgo):

    def __init__(self, discount_popularity=False, 
                 min_nn=1, 
                 max_nn=50,
                 min_threshold=None,
                 *args, **kwargs):
        self.discount_popularity = discount_popularity        
        self.min_nn = min_nn 
        self.max_nn = max_nn 
        self.min_threshold = min_threshold

        self.item_sim_matrix = None
        super(ItemCF, self).__init__(*args, **kwargs)

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
        rmat = self.rmat
        #normalize by item std
        item_norm = np.linalg.norm(rmat.toarray(), axis=0)
        nmat = rmat.toarray() / item_norm.reshape(1, -1)
        item_sim = nmat.T.dot(nmat)
        return item_sim

    def predict_for_user(self, user, items=None, ratings=None):
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

        for item in items:
            ipos = self.items.get_loc(item)

            # idx with decending similarities with itself
            item_idx = np.argsort(self.item_sim_matrix[ipos, :])[::-1][1:]

            # sim need to meet min_threshold
            if min_threshold is not None:
                item_idx = item_idx[self.item_sim_matrix[ipos, item_idx] > min_threshold]

            # narrow down to items were rated
            item_idx = item_idx[rmat_array[upos, item_idx] != 0]

            item_idx = item_idx[:max_nn]
            if len(item_idx) < min_nn:
                self.log.warning("item %s does not have enough neighbors (%s < %s)", item, len(item_idx), min_nn)
                result[item] = np.nan
                continue

            ratings = rmat_array[upos, item_idx]
            sim_wt = self.item_sim_matrix[ipos, item_idx]
            rating = ratings.dot(sim_wt) / sim_wt.sum()
            result[item] = rating

        df = pd.Series(result)
        return df


if __name__ == '__main__':

    ratings, users, movies = load_movielen_data()
    itemcf = ItemCF(min_threshold=0.1, min_nn=5)
    print(itemcf.get_params())
    itemcf.fit(ratings)
    user = 1
    movies = list(movies.item.astype(int))
    df = itemcf.predict_for_user(user, movies)
    print(df)