from abc import abstractmethod, ABC


from main.algorithm.basealgo import BaseAlgo
from main.algorithm.itemcf import ItemCF
from main.util.movielen_reader import load_movielen_data


class TopN(BaseAlgo, ABC):

    def __init__(self, predictor, selector=None):
        self.predictor = predictor
        self.selector = selector

    def fit(self, data, **kwargs):
        self.predictor.fit(data,**kwargs)
        if self.selector is not None:
            self.selector.fit(data, **kwargs)

    def recommend(self, user, n=None, candidates=None, ratings=None):

        if self.selector is not None:
            candidates = self.selector.select(user, candidates)

        scores = self.predictor.predict_for_user(user, candidates)
        scores = scores[scores.notna()].sort_values(ascending=False)

        if n is not None:
            scores = scores[:n]

        return scores


if __name__ == '__main__':
    ratings, users, movies = load_movielen_data()

    itemcf = ItemCF(min_threshold=0.1, min_nn=5, max_nn=20)
    user = 1
    obj = TopN(itemcf)
    obj.fit(ratings)
    print(obj.recommend(user, 50))
