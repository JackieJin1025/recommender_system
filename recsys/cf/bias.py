from recsys.cf.basic import Predictor
import numpy as np
import pandas as pd

from recsys.utils.data import load_movielen_data
from recsys.utils.debug import LogUtil
from recsys.utils.functions import _demean


class Bias(Predictor):
    """
        base line model that predict rating based one global_mean + user_bias + item_bias
        method: simply decouple global_mean, user_bias and item_bias from rating matrix
        Alternatively, we can use stochastic gradient descent to estimate global_mean, user_bias, and item_bias

    """

    def __init__(self, user_bias=True, item_bias=True, user_damping=10, item_damping=25):
        self.user_bias = user_bias
        self.item_bias = item_bias
        self.user_damping = user_damping
        self.item_damping = item_damping
        self.user_offset = None
        self.item_offset = None
        self.mean = None
        self.nmat = None
        super(Bias, self).__init__()

    def _train(self):
        m, n = self.rmat.shape
        self.user_offset = np.zeros(m)
        self.item_offset = np.zeros(n)

        rmat = self.rmat.tocsr(copy=True)
        self.mean = np.mean(rmat.data)
        rmat.data = rmat.data - self.mean
        if self.item_bias:
            # demean item
            rmat = rmat.tocsc()
            self.item_offset = _demean(rmat, self.item_damping)
        if self.user_bias:
            # demean user
            rmat = rmat.tocsr()
            self.user_offset = _demean(rmat, self.user_damping)

        rmat = rmat.tocsr()
        self.nmat = rmat

    def get_unbiased_rmat(self):
        return self.nmat

    def get_user_bias(self):
        return self.user_offset

    def get_item_bias(self):
        return self.item_offset

    def get_global_mean(self):
        return self.mean

    @property
    def pred(self):
        pred = self.user_offset.reshape(-1, 1) + self.item_offset.reshape(1, -1) + self.mean
        return pred

    def predict_for_user(self, user, items=None, ratings=None):
        if items is not None:
            items = np.array(items)
        else:
            items = self.items.values

        uidx = self.users.get_loc(user)
        pred = self.mean + self.user_offset[uidx] + self.item_offset

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
    LogUtil.configLog()
    ratings, users, movies = load_movielen_data()
    model = Bias()
    model.fit(ratings)
    item_bias = model.get_item_bias()
    user_bias = model.get_user_bias()
    print(item_bias)
    print(user_bias)
    print("item bias, avg %.3f, std %.3f" % (np.mean(item_bias), np.std(item_bias)))
    print("user bias, avg %.3f, std %.3f" % (np.mean(user_bias), np.std(user_bias)))