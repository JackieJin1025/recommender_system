from abc import abstractmethod
import os

from main.collaborative_filtering.itemcf import ItemCF
from main.util.data import get_data


class BaseRecommender:

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def recommend(self, user, n=None, candidates=None, ratings=None):
        pass



class TopN(BaseRecommender):

    def __init__(self, predictor, selector=None):
        self.predictor = predictor
        self.selector = selector


    def fit(self, data, **kwargs):
        self.predictor.fit(data,**kwargs)
        if self.selector is not None:
            self.selector.fit(data, **kwargs)


    def recommend(self, user, n=None, candidates=None, ratings=None):
        scores = self.predictor.predict_for_user(user, candidates)
        scores = scores[scores.notna()].sort_values(ascending=False)

        if n is not None:
            scores = scores[:n]

        return scores


if __name__ == '__main__':
    base_dir = "/Users/Jackie/Work/RecommendationSystem/data/ml-1m"
    movies = get_data(os.path.join(base_dir, "movies.dat"), 'MovieID::Title::Genres'.split("::"))
    ratings = get_data(os.path.join(base_dir, "ratings.dat") , "UserID::MovieID::Rating::Timestamp".split("::"))
    ratings = ratings.rename(columns={'UserID': 'user', 'MovieID': 'item', 'Rating': 'rating'})
    ratings[['user', 'item']] = ratings[['user', 'item']].astype(int)
    ratings['rating'] = ratings['rating'].astype(float)
    itemcf = ItemCF(min_threshold=0.1, min_nn=5)

    user = 1
    obj = TopN(itemcf)
    obj.fit(ratings)
    print(obj.recommend(user, 50))
