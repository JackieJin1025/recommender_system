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
from sklearn.decomposition import TruncatedSVD


class SVD(BaseAlgo):

    def __init__(self, n_iter, n_factor, *args, **kwargs):
        self.n_iter = n_iter
        self.n_factor = n_factor
        self.user_components = None
        self.svd = TruncatedSVD(n_components=n_factor, n_iter=n_iter, random_state=0)

        super(SVD, self).__init__(*args, **kwargs)

    def _train(self):
        rmat = self.rmat.toarray()

        # U, Sigma, VT = svd(rmat) ==> user_components = U.dot(Sigma)
        self.user_components = self.svd.fit_transform(rmat)

    def predict_for_user(self, user, items=None, ratings=None):
        # convert rmat to array
        if items is not None:
            items = np.array(items)
        else:
            items = self.items.values

        uidx = self.users.get_loc(user)
        Xt = self.user_components[[uidx], :]
        # Transform Xt back to its original space. Xt.dot(Vt)
        pred = self.svd.inverse_transform(Xt)
        pred = pred.flatten()

        # mark items not in the model universe as np.nan
        item_idx = self.items.get_indexer(items)
        invalid_items = items[item_idx == -1]
        df = pd.Series(data=np.NAN, index=invalid_items)
        item_idx = item_idx[item_idx >= 0]
        items = items[item_idx]
        pred = pred[item_idx]
        df = df.append(pd.Series(data=pred, index=items))
        return df


if __name__ == '__main__':

    ratings, users, movies = load_movielen_data()
    als = SVD(n_iter=40, n_factor=20)
    print(als.get_params())
    als.fit(ratings)
    user = 1
    movies = list(movies.item.astype(int))
    df = als.predict_for_user(user, movies)
    print(df)