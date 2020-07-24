import os
from main.algorithm.itemcf import ItemCF
from main.algorithm.lfm import LFM
from main.algorithm.usercf import UserCF
from main.algorithm.recommender import TopN
from main.algorithm.selector import NotRatedSelector
from main.util.data import get_data, train_test_split
from main.util.debug import LogUtil

from main.util.movielen_reader import load_movielen_data


def _test_item_cf():
    ratings, users, movies = load_movielen_data()
    predictor = ItemCF(min_threshold=0.1, min_nn=5, max_nn=20)
    selector = NotRatedSelector()
    user = 1
    model = TopN(predictor, selector)
    model.fit(ratings)
    print(model.recommend(user, 50))


if __name__ == '__main__':
    # _test_user_cf()
    _test_item_cf()