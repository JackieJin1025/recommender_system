from collections import defaultdict
from operator import itemgetter
import pickle
from sys import path

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from recommender.algorithm.basic import Predictor
from recommender.algorithm.bias import Bias
from recommender.utils.data import get_data, load_movielen_data
import os
import pandas as pd

from sklearn.decomposition import TruncatedSVD

from recommender.utils.debug import Timer, LogUtil
from recommender.utils.functions import scores_to_series


class BiasedSVD(Predictor):

    """
        Biased matrix factorization for implicit feedback using SciKit-Learn's SVD
        solver (:class:`sklearn.decomposition.TruncatedSVD`).  It operates by first
        computing the bias, then computing the SVD of the bias residuals.
    """

    def __init__(self, n_iter, n_factor=30, bias=Bias(), *args, **kwargs):
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
        super(BiasedSVD, self).fit(origin_data)
        return self

    def _train(self):
        rmat = self.rmat
        if self.bias is not None:
            rmat = self.bias.get_unbiased_rmat()

        rmat = rmat.toarray()
        # U, Sigma, VT = svd(rmat) ==> user_components = U.dot(Sigma)
        self.user_components = self.svd.fit_transform(rmat)

    def predict_for_user(self, user, items=None, ratings=None):
        if items is not None:
            items = np.array(items)
        else:
            items = self.items.values
        clock = Timer()
        uidx = self.users.get_loc(user)
        Xt = self.user_components[[uidx], :]
        # Transform Xt back to its original space. Xt.dot(Vt)
        pred = self.svd.inverse_transform(Xt)
        e0 = clock.restart()
        pred = pred.flatten()
        e1 = clock.restart()
        df = scores_to_series(pred, self.items, items)
        e2 = clock.restart()
        if self.bias is not None:
            bias_scores = self.bias.predict_for_user(user, items)
            df += bias_scores
        e3 = clock.restart()
        # print(e0, e1, e2, e3)
        return df


if __name__ == '__main__':

    LogUtil.configLog()

    ratings, users, movies = load_movielen_data()
    bias = Bias()
    model = BiasedSVD(n_iter=40, n_factor=20, bias=bias)
    print(model.get_params())
    model.fit(ratings)
    user = 1
    movies = list(movies.item.astype(int))
    clock = Timer()
    for i in range(10):
        df = model.predict_for_user(user, movies)
        # print(clock.restart())

    print(df.describe())
