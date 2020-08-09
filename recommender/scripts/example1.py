from recommender.algorithm.als import ExplicitALS
from recommender.algorithm.bias import Bias

from recommender.algorithm.itemcf import ItemCF
from recommender.algorithm.svd import BiasedSVD
from recommender.algorithm.usercf import UserCF
from recommender.algorithm.recommender import TopN
from recommender.algorithm.selector import BaseSelector
from recommender.algorithm.funksvd import FunkSVD
from recommender.utils.data import get_data, train_test_split, load_movielen_data
from recommender.utils.debug import LogUtil, Timer
import numpy as np


def _testRecommender():
    ratings, users, movies = load_movielen_data()
    predictor = ItemCF(min_threshold=0.1, min_nn=5, max_nn=20)
    selector = BaseSelector()
    user = 1
    model = TopN(predictor, selector)
    model.fit(ratings)
    print(model.recommend(user, 50))


def _testExplictALS():
    model = ExplicitALS(n_factors=40, n_iters=30, reg=0.001)
    ratings, users, movies = load_movielen_data()

    training, testing = train_test_split(ratings)
    model.fit(training)
    model.eval(testing)


def _testFunkSVD():
    model = FunkSVD(learning_rate=0.001, reg=0.005, n_epochs=100, n_factors=30)
    ratings, users, movies = load_movielen_data()

    training, testing = train_test_split(ratings)
    model.fit(training)
    model.eval(testing)


def _testItemCF():
    bias = Bias()
    model = ItemCF(min_threshold=0.1, min_nn=1, max_nn=30, bias=bias)
    ratings, users, movies = load_movielen_data()

    training, testing = train_test_split(ratings)
    model.fit(training)
    model.eval(testing)


def _testUserCF():
    model = UserCF(min_threshold=0.1, min_nn=1, max_nn=30)
    ratings, users, movies = load_movielen_data()
    training, testing = train_test_split(ratings)
    model.fit(training)
    model.eval(testing)


def _testBiasedSVD():
    bias = Bias()
    # bias = None
    model = BiasedSVD(n_iter=40, n_factor=20, bias=bias)
    ratings, users, movies = load_movielen_data()
    training, testing = train_test_split(ratings)
    model.fit(training)
    model.eval(testing)


if __name__ == '__main__':
    LogUtil.configLog()
    # _testUserCF()
    # _testItemCF()
    _testFunkSVD()
    # _testBiasedSVD()
    # _testExplictALS()