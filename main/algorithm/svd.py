from collections import defaultdict
from operator import itemgetter
import pickle
from sys import path

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from main.algorithm.basic import Predictor
from main.algorithm.bias import Bias
from main.util.data import get_data, load_movielen_data
import os
import pandas as pd

from sklearn.decomposition import TruncatedSVD

from main.util.debug import Timer


class BiasedSVD(Predictor):

    """
        Biased matrix factorization for implicit feedback using SciKit-Learn's SVD
        solver (:class:`sklearn.decomposition.TruncatedSVD`).  It operates by first
        computing the bias, then computing the SVD of the bias residuals.
    """

    def __init__(self, n_iter, n_factor, bias=None, *args, **kwargs):
        self.n_iter = n_iter
        self.n_factor = n_factor
        self.user_components = None
        self.svd = TruncatedSVD(n_components=n_factor, n_iter=n_iter, random_state=0)
        self.bias = bias

        super(BiasedSVD, self).__init__(*args, **kwargs)

    def fit(self, origin_data):
        """
            to call bias.fit()
        """
        if self.bias is not None:
            self.bias.fit(origin_data)

        clock = Timer()
        self.process_data(origin_data)
        e0 = clock.restart()
        self.log.info('loading init data takes %.3f ...', e0)
        flag = self.filename is not None
        if flag and path.exists(self.filename):
            self._load()
            return

        _ = clock.restart()
        self.log.info('start training ...')
        self._train()
        e2 = clock.restart()
        self.log.info('training takes %.3f ...', e2 )
        if flag:
            self._save()
        return self

    def _train(self):
        rmat = self.rmat.toarray()
        if self.bias is not None:
            bias_pred = self.bias.pred
            rmat = rmat - bias_pred

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

        if self.bias is not None:
            pred += self.bias.pred[uidx, :]

        # mark items not in the model universe as np.nan
        item_idx = self.items.get_indexer(items)
        invalid_items = items[item_idx == -1]
        df = pd.Series(data=np.NAN, index=invalid_items)
        item_idx = item_idx[item_idx >= 0]
        items = self.items[item_idx]
        pred = pred[item_idx]
        df = df.append(pd.Series(data=pred, index=items))
        return df


if __name__ == '__main__':

    ratings, users, movies = load_movielen_data()
    bias = Bias()
    model = BiasedSVD(n_iter=40, n_factor=20, bias=bias)
    print(model.get_params())
    model.fit(ratings)
    user = 1
    movies = list(movies.item.astype(int))
    df = model.predict_for_user(user, movies)
    print(df.describe())

"""
count    3706.000000
mean        0.059754
std         0.266182
min        -0.512490
25%        -0.028333
50%         0.003629
75%         0.044354
max         3.059997
"""