from collections import defaultdict
from operator import itemgetter
import pickle

from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from main.algorithm.basic import Predictor
from main.util.data import get_data, load_movielen_data
import os
import pandas as pd

from main.util.metric import RMSE


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
    def __init__(self, n_iters, n_factors, reg, *args, **kwargs):
        self.n_iters = n_iters
        self.n_factors = n_factors
        self.reg = reg
        self.user_p = None
        self.item_q = None
        super(ExplicitALS, self).__init__(*args, **kwargs)

    def _init_latent_factor(self):
        """
            initialize latent matrix
        """
        k = self.n_factors
        m = len(self.users)
        n = len(self.items)

        # initilize latent factor matrix for users and items
        self.user_p = np.random.normal(size=(m, k))
        self.item_q = np.random.normal(size=(n, k))

    def _train(self):
        self._init_latent_factor()

        y_true = self.rmat.toarray()

        for i in range(self.n_iters):
            self.user_p = self._als_step(self.rmat, self.item_q)
            self.item_q = self._als_step(self.rmat.T, self.user_p)
            y_pred = self._predict()

            mask = np.nonzero(y_true)
            err = RMSE(y_pred[mask], y_true[mask])
            if i % 5 == 0:
                self.log.info("RMSE on iter %d : %.3f", i, err)

    def _als_step(self, r, Y):
        """
        r: ratings with m x n  and Y is n x k

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
        r = r.toarray()
        I = np.eye(self.n_factors)

        A = Y.T.dot(Y) + I * self.reg  # k x k
        A_inv = np.linalg.inv(A)
        X = r.dot(Y).dot(A_inv)
        return X


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
        item_idx = self.items.get_indexer(items)
        invalid_items = items[item_idx == -1]
        df = pd.Series(data=np.NAN, index=invalid_items)

        item_idx = item_idx[item_idx >= 0]
        pred = self._predict(user)
        items = items[item_idx]
        pred = pred[item_idx]
        df = df.append(pd.Series(data=pred, index=items))
        return df


if __name__ == '__main__':

    ratings, users, movies = load_movielen_data()
    als = ExplicitALS(n_factors=40, n_iters=20, reg=0.001)
    print(als.get_params())
    als.fit(ratings)
    user = 1
    movies = list(movies.item.astype(int))
    df = als.predict_for_user(user, movies)
    print(df)
    print(df.describe())