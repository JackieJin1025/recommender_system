from main.algorithm.basic import Predictor
import numpy as np
import pandas as pd

from main.utils.data import load_movielen_data
from main.utils.debug import LogUtil, timer, Timer
from main.utils.functions import _demean


class Bias(Predictor):

    def __init__(self,  *args, **kwargs):
        self.user_offset = None
        self.item_offset = None
        self.mean = None
        self.pred = None
        self.nmat = None
        super(Bias, self).__init__(*args, **kwargs)

    def _train(self):
        rmat = self.rmat.tocsr()
        self.mean = np.mean(rmat.data)
        rmat.data = rmat.data - self.mean
        # demean item
        rmat = rmat.tocsc()
        self.item_offset = _demean(rmat)
        # demean user
        rmat = rmat.tocsr()
        self.user_offset = _demean(rmat)

        self.pred = self.user_offset.reshape(-1, 1) + self.item_offset.reshape(1, -1) + self.mean
        self.nmat = rmat

    def get_user_bias(self):
        return self.user_offset

    def get_item_bias(self):
        return self.item_offset

    def predict_for_user(self, user, items=None, ratings=None):
        # convert rmat to array
        if items is not None:
            items = np.array(items)
        else:
            items = self.items.values

        uidx = self.users.get_loc(user)
        pred = self.pred[uidx, :]

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
    LogUtil.configLog()
    ratings, users, movies = load_movielen_data()
    model = Bias()
    model.fit(ratings)
    print(model.pred)
    # user = 1
    # movies = list(movies.item.astype(int))
    # df = model.predict_for_user(user, movies)
    # print(df)