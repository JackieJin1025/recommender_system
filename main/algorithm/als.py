
import numpy as np
from numba import njit
from main.algorithm.basic import Predictor
from main.algorithm.bias import Bias
from main.utils.data import load_movielen_data
import os
import pandas as pd

from main.utils.debug import LogUtil
from main.utils.functions import scores_to_series
from main.utils.metric import _evaluate


def _als_step(rmat, Y, n_factors, reg):
    """
    rmat: ratings with m x n  and Y is n x k

    implicit:
    x_u = (Y.T*Y + Y.T*(Cu - I) * Y + lambda*I)^-1 * (X.T * Cu * p(u))
    y_i = (X.T*X + X.T*(Ci - I) * X + lambda*I)^-1 * (Y.T * Ci * p(i))
    p(u), p(i) : preference with value in {0, 1}  p = 1 if r > 0 else 0.
    Cu, Ci: confidence: i.e.  c = 1 + alpha * r

    explicit:
    x_u = (Y.T*Y  + lambda*I)^-1 * (X.T *  r(u))
    y_i = (X.T*X  + lambda*I)^-1 * (Y.T *  r(i))
    r: scores
    """
    I = np.eye(n_factors)
    A = Y.T.dot(Y) + I * reg  # k x k
    A_inv = np.linalg.inv(A)
    XR = rmat.dot(Y)
    X = XR.dot(A_inv)
    return X


class ExplicitALS(Predictor):
    """
    Train a matrix factorization model using explicit Alternating Least Squares
    to predict empty entries in a matrix

    reference:
    https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe
    http://yifanhu.net/PUB/cf.pdf
    http://ethen8181.github.io/machine-learning/recsys/1_ALSWR.html
    https://www.ethanrosenthal.com/2016/01/09/explicit-matrix-factorization-sgd-als/
    https://math.stackexchange.com/questions/1072451/analytic-solution-for-matrix-factorization-using-alternating-least-squares

    """
    def __init__(self, n_iters=30, n_factors=20, reg=0.001, bias=None, *args, **kwargs):
        self.n_iters = n_iters
        self.n_factors = n_factors
        self.reg = reg
        self.user_p = None
        self.item_q = None
        self.bias = bias
        super(ExplicitALS, self).__init__(*args, **kwargs)

    def _init_latent_factor(self):
        """
            initialize latent matrix
        """
        k = self.n_factors
        m = len(self.users)
        n = len(self.items)

        # initilize latent factor matrix for users and items
        self.user_p = np.random.normal(0, 0.1, size=(m, k))
        self.item_q = np.random.normal(0, 0.1, size=(n, k))

    def fit(self, data):
        if self.bias is not None:
            self.bias.fit(data)
        super(ExplicitALS, self).fit(data)

    def _train(self):
        self._init_latent_factor()
        rmat = self.rmat

        global_mean, bu, bi = None, None, None
        if self.bias is not None:
            rmat = self.bias.get_unbiased_rmat()
            global_mean = self.bias.get_global_mean()
            bu = self.bias.get_user_bias()
            bi = self.bias.get_item_bias()

        rmat = rmat.toarray()
        for i in range(self.n_iters):
            self.user_p = _als_step(rmat, self.item_q, self.n_factors, self.reg)
            self.item_q = _als_step(rmat.T, self.user_p, self.n_factors, self.reg)
            # y_pred = self._predict()
            rmse, mae = _evaluate(rmat, self.user_p, self.item_q, bu, bi, global_mean)

            if i % 5 == 0:
                self.log.info('Epoch {}/{}, rmse {}, mae {}'.format(i + 1, self.n_iters, rmse, mae))

    def _predict(self, user=None):

        if user is None:
            pred = self.user_p.dot(self.item_q.T)
            return pred

        idx = self.users.get_loc(user)
        pred = self.item_q.dot(self.user_p[idx, :])
        return pred

    def predict_for_user(self, user, items=None, ratings=None):

        # convert rmat to array
        if items is not None:
            items = np.array(items)
        else:
            items = self.items.values

        # mark items not in the model universe as np.nan
        pred = self._predict(user)
        df = scores_to_series(pred, self.items, items)
        if self.bias is not None:
            df += self.bias.predict_for_user(user, items)

        return df


if __name__ == '__main__':
    LogUtil.configLog()
    ratings, users, movies = load_movielen_data()
    bias = Bias()
    als = ExplicitALS(n_factors=40, n_iters=30, reg=0.001, bias=bias)
    print(als.get_params())
    als.fit(ratings)
    user = 1
    movies = list(movies.item.astype(int))
    df = als.predict_for_user(user, movies)
    print(df)
    print(df.describe())