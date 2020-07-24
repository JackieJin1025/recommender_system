from abc import ABC

import numpy as np
from main.algorithm.basealgo import BaseAlgo
from main.util.movielen_reader import load_movielen_data


class NotRatedSelector(BaseAlgo, ABC):
    def __init__(self, **kwargs):
        super(NotRatedSelector, self).__init__(**kwargs)

    def _train(self):
        pass

    def _save(self):
        pass

    def _load(self):
        pass

    def select(self, user, candidates=None):
        """
        :param user: user id
        :param candidates: a list or np.array of items
        :return: return items not reviewed by user id
        """
        upos = self.users.get_loc(user)
        ratings = self.rmat.toarray()[upos, :]
        idx = np.argwhere(ratings==0).flatten()
        if candidates is not None:
            candidates = np.array(candidates)
            base = self.items.get_indexer(candidates)
            idx = np.intersect1d(idx, base)

        items = self.items[idx]
        items = np.array(items)
        return items


if __name__ == '__main__':
    ratings, users, movies = load_movielen_data()
    model = NotRatedSelector()
    model.fit(ratings)
    print(model.select(1, [2, 3, 4]))