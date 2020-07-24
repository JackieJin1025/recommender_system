import pandas as pd

class EqualAggregator:
    def __init__(self):
        pass

    def combine(self, scores):
        """

        :param scores: a list of pd.series
        :return: pd.series
        """
        df = pd.concat(scores, axis=1)
        result = df.mean(axis=1)
        return result