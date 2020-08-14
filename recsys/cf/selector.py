from abc import ABC, abstractmethod

import numpy as np
from recsys.cf.basic import Predictor, BaseAlgo
from recsys.utils.data import load_movielen_data, sparse_ratings
from recsys.utils.debug import LogUtil
from recsys.utils.functions import _get_xs
from recsys.utils.metric import sparsity


class BaseSelector(BaseAlgo):
    def __init__(self, **kwargs):
        self.rmat = None
        self.users = None
        self.items = None
        self.log = LogUtil.getLogger(self.__class__.__name__)
        super(BaseSelector, self).__init__(**kwargs)

    def fit(self, origin_data):
        self.rmat, self.users, self.items = sparse_ratings(origin_data)
        spar = sparsity(self.rmat)
        self.log.info("%d users %d items with sparsity %.2f", self.rmat.shape[0], self.rmat.shape[1], spar)

    @abstractmethod
    def select(self, user, candidates=None):
        raise NotImplementedError()


class NotRatedSelector(BaseSelector):
    def __init__(self, **kwargs):
        super(NotRatedSelector, self).__init__(**kwargs)

    def select(self, user, candidates=None):
        """
        :param user: user id
        :param candidates: a list or np.array of items
        :return: return items not reviewed by user id
        """
        upos = self.users.get_loc(user)
        ratings = _get_xs(self.rmat, upos)
        idx = np.argwhere(ratings == 0).flatten()
        if candidates is not None:
            candidates = np.array(candidates)
            base = self.items.get_indexer(candidates)
            idx = np.intersect1d(idx, base)

        items = self.items[idx]
        items = np.array(items)
        return items


class RatedSelector(BaseSelector):
    def __init__(self, **kwargs):
        super(RatedSelector, self).__init__(**kwargs)

    def select(self, user, candidates=None):
        """
        :param user: user id
        :param candidates: a list or np.array of items
        :return: return items not reviewed by user id
        """
        upos = self.users.get_loc(user)
        ratings = _get_xs(self.rmat, upos)
        idx = np.argwhere(ratings != 0).flatten()
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
    print(model.select(1))

    model1 = RatedSelector()
    model1.fit(ratings)
    print(model1.select(1))