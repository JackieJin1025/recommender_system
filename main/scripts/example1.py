import os

from numba import njit

from main.algorithm.bias import Bias
from main.algorithm.funksvd import FunkSVD
from main.algorithm.itemcf import ItemCF
from main.algorithm.lfm import LFM
from main.algorithm.svd import BiasedSVD
from main.algorithm.usercf import UserCF
from main.algorithm.recommender import TopN
from main.algorithm.selector import NotRatedSelector
from main.utils.data import get_data, train_test_split, load_movielen_data
from main.utils.debug import LogUtil, Timer
import numpy as np


def _test_item_cf():
    ratings, users, movies = load_movielen_data()
    predictor = ItemCF(min_threshold=0.1, min_nn=5, max_nn=20)
    selector = NotRatedSelector()
    user = 1
    model = TopN(predictor, selector)
    model.fit(ratings)
    print(model.recommend(user, 50))


def _testFunkSVD():
    model = FunkSVD(learning_rate=0.001, reg=0.005, n_epochs=100, n_factors=30)
    ratings, users, movies = load_movielen_data()

    training, testing = train_test_split(ratings)
    model.fit(training)
    model.eval(testing)


def _testItemCF():
    model = ItemCF(min_threshold=0.1, min_nn=1, max_nn=20)
    ratings, users, movies = load_movielen_data()

    training, testing = train_test_split(ratings)
    model.fit(training)
    model.eval(testing)


def _testUserCF():
    model = UserCF(min_threshold=0.1, min_nn=1)
    ratings, users, movies = load_movielen_data()

    training, testing = train_test_split(ratings)
    model.fit(training)
    model.eval(testing)

def _testBiasedSVD():
    bias = Bias()
    model = BiasedSVD(n_iter=40, n_factor=20, bias=bias)
    ratings, users, movies = load_movielen_data()
    training, testing = train_test_split(ratings)
    model.fit(training)
    model.eval(testing)
    from sklearn.decomposition import TruncatedSVD, PCA


if __name__ == '__main__':
    # _test_user_cf()
    _testUserCF()