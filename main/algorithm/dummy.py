import numpy as np

from main.algorithm.basic import Predictor
from main.utils.functions import _get_xs, scores_to_series


class DummyPredictor(Predictor):
    """
        return the existing scores
    """

    """
    """
    def __init__(self, *args, **kwargs):
        super(DummyPredictor, self).__init__(*args, **kwargs)

    def _train(self):
        pass

    def predict_for_user(self, user, items=None, ratings=None):
        uidx = self.users.get_loc(user)
        scores = _get_xs(self.rmat, uidx)
        scores[scores == 0] = np.nan

        if items is not None:
            items = np.array(items)
        else:
            items = self.items.values
        df = scores_to_series(scores, self.items, items)
        return df