import os
from main.collaborative_filtering.itemcf import ItemCF
from main.collaborative_filtering.lfm import LFM
from main.collaborative_filtering.usercf import UserCF
from main.util.data import get_data, train_test_split
from main.util.debug import LogUtil



def _test_item_cf():
    base_dir = "/Users/Jackie/Work/RecommendationSystem/data/ml-1m"
    movies = get_data(os.path.join(base_dir, "movies.dat"), 'MovieID::Title::Genres'.split("::"))
    ratings = get_data(os.path.join(base_dir, "ratings.dat") , "UserID::MovieID::Rating::Timestamp".split("::"))
    users = get_data(os.path.join(base_dir, "users.dat"), "UserID::Gender::Age::Occupation::Zip-code".split("::"))
    user_ids = list(users['UserID'].unique())
    train_data, test_data = train_test_split(ratings[['UserID', 'MovieID', 'Rating']], frac=0.2)

    cached_filename = "/Users/Jackie/Work/RecommendationSystem/data/item_cf.pickle"
    model = ItemCF(discount_popularity=True, filename=cached_filename)
    model.train(train_data)
    model.test(None, 10, test_data)


def _test_user_cf():
    base_dir = "/Users/Jackie/Work/RecommendationSystem/data/ml-1m"
    movies = get_data(os.path.join(base_dir, "movies.dat"), 'MovieID::Title::Genres'.split("::"))
    ratings = get_data(os.path.join(base_dir, "ratings.dat") , "UserID::MovieID::Rating::Timestamp".split("::"))
    users = get_data(os.path.join(base_dir, "users.dat"), "UserID::Gender::Age::Occupation::Zip-code".split("::"))
    user_ids = list(users['UserID'].unique())
    train_data, test_data = train_test_split(ratings[['UserID', 'MovieID', 'Rating']], frac=0.2)

    cached_filename = "/Users/Jackie/Work/RecommendationSystem/data/user_cf.pickle"
    model = UserCF(discount_popularity=True, filename=cached_filename)
    model.train(train_data)
    model.test(None, 10, test_data)


if __name__ == '__main__':
    _test_user_cf()
    _test_item_cf()