from recsys.cf.basic import Predictor
import numpy as np
import pandas as pd

from recsys.utils.data import load_movielen_data
from recsys.utils.debug import LogUtil, timer, Timer
from recsys.utils.functions import _demean


class Bias(Predictor):

    def __init__(self,  user_bias=True, item_bias=True, *args, **kwargs):
        self.user_bias = user_bias
        self.item_bias = item_bias
        self.user_offset = None
        self.item_offset = None
        self.mean = None
        self.nmat = None

        super(Bias, self).__init__(*args, **kwargs)

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
            self.item_offset = _demean(rmat)
        if self.user_bias:
            # demean user
            rmat = rmat.tocsr()
            self.user_offset = _demean(rmat)

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
    model = Bias(user_bias=False)
    model.fit(ratings)
    print(model.get_item_bias())
    print(model.get_user_bias())
    user = 1
    movies = list(movies.item.astype(int))
    df = model.predict_for_user(user, movies)
    print(df)