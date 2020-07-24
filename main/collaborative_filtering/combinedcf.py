from main.collaborative_filtering.aggregator import EqualAggregator
from main.collaborative_filtering.basecf import BaseCF


class CombinedCF(BaseCF):
    def __init__(self, predictors, aggregator=None, *args, **kwargs):
        self.predictors = predictors
        if aggregator is None:
            aggregator = EqualAggregator()
        self.aggregator = aggregator
        super(CombinedCF, self).__init__(*args, **kwargs)

    def fit(self, origin_data):
        """
            train model
        """
        for predictor in self.predictors:
            predictor.fit(origin_data)


    def predict_for_user(self, user, items, ratings=None):
        scores = [predictor.predict_for_user(user, items, ratings) for predictor in self.predictors]
        score = self.aggregator.combine(scores)
        return score
