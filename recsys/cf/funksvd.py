from numba import njit

from recsys.cf.basic import Predictor
import pandas as pd
import numpy as np

from recsys.utils.data import load_movielen_data, train_test_split
from recsys.utils.debug import timer, LogUtil
from recsys.utils.functions import scores_to_series
from recsys.utils.metric import MAE, _evaluate


@njit
def _shuffle(X):
    np.random.shuffle(X)
    return X


@njit
def _run_epoch(X, pu, qi, bu, bi, global_mean,  lr, reg):
    """Runs an epoch, updating model weights (pu, qi, bu, bi).

    Args:
        X (numpy array): training set.
        pu (numpy array): users latent factor matrix.
        qi (numpy array): items latent factor matrix.
        bu (numpy array): users biases vector.
        bi (numpy array): items biases vector.
        global_mean (float): ratings arithmetic mean.
        lr (float): learning rate.
        reg (float): regularization factor.

    Returns:
        pu (numpy array): users latent factor matrix updated.
        qi (numpy array): items latent factor matrix updated.
        bu (numpy array): users biases vector updated.
        bi (numpy array): items biases vector updated.
    """

    n_factors = pu.shape[1]

    for i in range(X.shape[0]):
        user, item, rating = int(X[i, 0]), int(X[i, 1]), X[i, 2]

        # Predict current rating
        pred = global_mean + bu[user] + bi[item]
        for factor in range(n_factors):
            pred += pu[user, factor] * qi[item, factor]

        err = rating - pred

        # Update biases
        bu[user] += lr * (err - reg * bu[user])
        bi[item] += lr * (err - reg * bi[item])

        # Update latent factors
        for factor in range(n_factors):
            puf = pu[user, factor]
            qif = qi[item, factor]

            pu[user, factor] += lr * (err * qif - reg * puf)
            qi[item, factor] += lr * (err * puf - reg * qif)

    return pu, qi, bu, bi


class FunkSVD(Predictor):
    """
        simon funk SVD with bias term
        reference:
        http://sifter.org/simon/journal/20061211.html
        https://github.com/gbolmier/funk-svd
    """
    def __init__(self, n_epochs, n_factors, learning_rate, reg, shuffle=False):
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.reg = reg
        self.lr = learning_rate
        self.user_p = None
        self.item_q = None
        self.shuffle = shuffle
        super(FunkSVD, self).__init__()

    def _init_data(self):
        """
            initialize latent matrix
        """
        k = self.n_factors
        m = len(self.users)
        n = len(self.items)

        # initilize latent factor matrix for users and items
        self.user_p = np.random.normal(0, 0.1, size=(m, k))
        self.item_q = np.random.normal(0, 0.1, size=(n, k))
        self.user_offsets = np.zeros(m)
        self.item_offsets = np.zeros(n)

        # reconstruct dense array
        rmat = self.rmat.tocoo()
        self.X = np.array([rmat.row, rmat.col, rmat.data]).T
        self.global_mean = np.mean(self.X[:, 2])

    def _train(self):
        self._init_data()
        self._sgd(self.X)

    def _sgd(self, X):
        """Performs SGD algorithm, learns model weights.

        Args:
            X (numpy array): training set, first column must contains users
                indexes, second one items indexes, and third one ratings.
            X_val (numpy array or `None`): validation set with same structure
                as X.
        """

        pu, qi, bu, bi = self.user_p, self.item_q, self.user_offsets, self.item_offsets

        # Run SGD
        for epoch_ix in range(self.n_epochs):
            if self.shuffle:
                X = _shuffle(X)

            pu, qi, bu, bi = _run_epoch(X, pu, qi, bu, bi, self.global_mean, self.lr, self.reg)

            if epoch_ix % 10 == 0:
                rmse, mae = _evaluate(self.rmat.toarray(), pu, qi, bu, bi, self.global_mean)
                self.log.info('Epoch {}/{}, rmse {}, mae {}'.format(epoch_ix + 1, self.n_epochs, rmse, mae))

        self.user_p = pu
        self.item_q = qi
        self.user_offsets = bu
        self.item_offsets = bi

    def predict_for_user(self, user, items=None, ratings=None):

        if items is not None:
            items = np.array(items)
        else:
            items = self.items.values

        uidx = self.users.get_loc(user)
        u_factor = self.user_p[uidx, :]
        u_mean = self.user_offsets[uidx]
        pred = self.item_q.dot(u_factor)
        pred += self.global_mean
        pred += u_mean
        pred += self.item_offsets
        df = scores_to_series(pred, self.items, items)
        return df


if __name__ == '__main__':
    LogUtil.configLog()
    model = FunkSVD(learning_rate=0.001, reg=0.005, n_epochs=100, n_factors=0)
    ratings, users, movies = load_movielen_data()
    model.fit(ratings)
    user = 1
    movies = list(movies.item.astype(int))
    df = model.predict_for_user(user, movies)
    print(df.describe())

