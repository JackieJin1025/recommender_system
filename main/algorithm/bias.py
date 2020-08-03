from main.algorithm.basic import Predictor
import numpy as np
import pandas as pd

from main.util.data import load_movielen_data


class Bias(Predictor):

    def __init__(self,  *args, **kwargs):
        self.user_offset = None
        self.item_offset = None
        self.mean = None
        self.pred = None
        super(Bias, self).__init__(*args, **kwargs)

    def _train(self):
        rmat = self.rmat.toarray()
        rmat[rmat == 0] = np.nan
        self.mean = np.nanmean(rmat)
        rmat = rmat - self.mean
        self.item_offset = np.nanmean(rmat, axis=0)
        # make sure average rating of each item is 0
        rmat = rmat - self.item_offset
        self.user_offset = np.nanmean(rmat, axis=1)
        self.pred = self.user_offset.reshape(-1, 1) + self.item_offset.reshape(1, -1) + self.mean

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
    ratings, users, movies = load_movielen_data()
    model = Bias()
    print(model.get_params())
    model.fit(ratings)
    user = 1
    movies = list(movies.item.astype(int))
    df = model.predict_for_user(user, movies)
    print(df)