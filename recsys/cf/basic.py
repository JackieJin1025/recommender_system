import inspect
from abc import abstractmethod
from collections import defaultdict

from sklearn.base import BaseEstimator
from tqdm import tqdm
from recsys.utils.data import sparse_ratings
from recsys.utils.debug import LogUtil, Timer, timer
from recsys.utils.metric import precision, recall, coverage, sparsity, RMSE, MAE, precision_recall_at_k
from os import path
import pandas as pd


class Predictor(BaseEstimator):
    def __init__(self, filename=None):
        """
            :param filename: if provided, cache parameters trained
        """
        self.filename = filename
        self.rmat = None
        self.users = None
        self.items = None
        self.log = LogUtil.getLogger(self.__class__.__name__)
        super(Predictor, self).__init__()

    def process_data(self, origin_data):
        """
        :param origin_data: dataframe 'user', 'item', 'rating'
        :return:
        """
        self.rmat, self.users, self.items = sparse_ratings(origin_data)
        spar = sparsity(self.rmat)
        self.log.info("%d users %d items with sparsity %.2f", self.rmat.shape[0], self.rmat.shape[1], spar)

    def fit(self, origin_data):
        clock = Timer()
        self.process_data(origin_data)
        e0 = clock.restart()
        self.log.info('loading init data takes %.3f secs...', e0)
        flag = self.filename is not None
        if flag and path.exists(self.filename):
            self._load()
            return

        _ = clock.restart()
        self.log.info('start training ...')
        self._train()
        e2 = clock.restart()
        self.log.info('training takes %.3f secs ...', e2 )
        if flag:
            self._save()
        return self

    @abstractmethod
    def _train(self):
        raise NotImplemented()

    @abstractmethod
    def predict_for_user(self, user, items=None, ratings=None):
        raise NotImplemented()

    def _load(self):
        pass

    def _save(self):
        pass

    def eval(self, x_val):
        """
        evaluate test data with RMSE and MAE
        :param x_val: is dataframe  with 'user', 'item', 'rating'
        :return: RMSE, MAE
        """
        clock = Timer()
        self.log.info("start evaluating with %d test samples ...", x_val.shape[0])
        group = x_val.groupby('user')
        df_summary = pd.DataFrame()
        for user, df in tqdm(group):
            actual = df[['item', 'rating']].set_index('item')['rating']
            pred = self.predict_for_user(user, items=actual.index)
            df_summary = df_summary.append(pd.DataFrame({'pred': pred,
                                  'actual': actual}))

        rmse = RMSE(df_summary.pred, df_summary.actual)
        mae = MAE(df_summary.pred, df_summary.actual)

        e0 = clock.restart()
        self.log.info("rmse: %.3f, mae: %.3f", rmse, mae)
        self.log.info("evaluation takes %.3f", e0)
        return rmse, mae



class Recommender(BaseEstimator):

    def __init__(self, *args, **kwargs):
        self(Recommender, self).__init__(*args, **kwargs)

    @abstractmethod
    def fit(self, data, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def recommend(self, user, n=None, candidates=None, ratings=None):
        raise NotImplementedError()

    def eval(self, x_val, N, threshold=3.5):
        """
        not finished yet !

        running test data and output metrics including precision, recall, coverage
        :param N: top N recommendations
        :param x_val:is dataframe 'UserID', 'MovieID', 'Rating'

        """

        # test dict: userId -> [movieID1, movieID2 ]
        tests = x_val.groupby('UserID')['MovieID'].apply(list).to_dict()
        all_items = list(x_val['MovieID'].unique())
        recommends = dict()

        cnt = 0
        for user_id, _ in self.user2idx.items():
            test_movies = tests.get(user_id, [])
            n = len(test_movies)
            # recommend the same number of movies watched by user_id
            recommends[user_id] = self.recommend(user_id, N)
            cnt += 1
            if cnt > 500:
                break

        prec = precision(recommends, tests, threshold=threshold)
        rec = recall(recommends, tests, threshold=threshold)
        cov = coverage(recommends, all_items)

        self.log.info("precision: %.3f", prec)
        self.log.info("recall: %.3f", rec)
        self.log.info("coverage: %.3f", cov)