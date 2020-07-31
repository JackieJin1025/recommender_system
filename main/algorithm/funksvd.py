from main.algorithm.basealgo import BaseAlgo
import pandas as pd
import numpy as np

from main.util.data import load_movielen_data
from main.util.fast_methods import _shuffle, _run_epoch


class FunkSVD(BaseAlgo):
    """
        simon funk SVD algorithm
        reference:
        http://sifter.org/simon/journal/20061211.html
        https://github.com/gbolmier/funk-svd
    """
    def __init__(self, n_epochs, n_factors, learning_rate, reg, *args, **kwargs):
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.reg = reg
        self.lr = learning_rate
        self.user_p = None
        self.item_q = None
        self.shuffle = False
        super(FunkSVD, self).__init__(*args, **kwargs)

    def _init_data(self):
        """
            initialize latent matrix
        """
        k = self.n_factors
        m = len(self.users)
        n = len(self.items)

        # initilize latent factor matrix for users and items
        self.user_p = np.random.normal(size=(m, k))
        self.item_q = np.random.normal(size=(n, k))
        self.user_offsets = np.random.normal(size=m)
        self.item_offsets = np.random.normal(size=n)

        # reconstruct dense array
        self.X = np.array([self.rmat.row, self.rmat.col, self.rmat.data]).T
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

            pu, qi, bu, bi = _run_epoch(X, pu, qi, bu, bi, self.global_mean,
                                        self.n_factors, self.lr, self.reg)

            if epoch_ix % 10 == 0:
                self.log.info('Epoch {}/{}'.format(epoch_ix + 1, self.n_epochs))

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
    model = FunkSVD(learning_rate=0.001, reg=0.005, n_epochs=100, n_factors=15)
    ratings, users, movies = load_movielen_data()
    model.fit(ratings)
    user = 1
    movies = list(movies.item.astype(int))
    df = model.predict_for_user(user, movies)
    print(df.describe())