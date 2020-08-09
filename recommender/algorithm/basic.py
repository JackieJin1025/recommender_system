import inspect
from abc import abstractmethod
from collections import defaultdict
from tqdm import tqdm
from recommender.utils.data import sparse_ratings
from recommender.utils.debug import LogUtil, Timer, timer
from recommender.utils.metric import precision, recall, coverage, sparsity, RMSE, MAE
from os import path
import pandas as pd

class BaseAlgo(object):
    def __init__(self):
        pass

    def get_params(self, deep=True):
        """
        Get the parameters for this algorithm (as in scikit-learn).  Algorithm parameters
        should match constructor argument names.

        The default implementation returns all attributes that match a constructor parameter
        name.  It should be compatible with :py:meth:`scikit.base.BaseEstimator.get_params`
        method so that LensKit alogrithms can be cloned with :py:func:`scikit.base.clone`
        as well as :py:func:`lenskit.util.clone`.

        Returns:
            dict: the algorithm parameters.
        """
        sig = inspect.signature(self.__class__)
        names = list(sig.parameters.keys())
        params = {}
        for name in names:
            if hasattr(self, name):
                value = getattr(self, name)
                params[name] = value
                if deep and hasattr(value, 'get_params'):
                    sps = value.get_params(deep)
                    for k, sv in sps.items():
                        params[name + '__' + k] = sv

        return params


class Predictor(BaseAlgo):
    def __init__(self, filename=None, *args, **kwargs):
        """
            :param filename: if provided, cache parameters trained
        """
        self.filename = filename
        self.rmat = None
        self.users = None
        self.items = None
        self.log = LogUtil.getLogger(self.__class__.__name__)
        super(Predictor, self).__init__(*args, **kwargs)

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

    def test(self, N, K, test_df):
        """
        running test data and output metrics including precision, recall, coverage
        :param N: top N recommendations
        :param k: k similar users/items
        :param test_df:is dataframe 'UserID', 'MovieID', 'Rating'
        """

        # test dict: userId -> [movieID1, movieID2 ]
        tests = test_df.groupby('UserID')['MovieID'].apply(list).to_dict()
        all_items = list(test_df['MovieID'].unique())
        recommends = dict()

        cnt = 0
        for user_id, _ in self.user2idx.items():
            test_movies = tests.get(user_id, [])
            n = len(test_movies)
            # recommend the same number of movies watched by user_id
            recommends[user_id] = self.recommend(user_id, n, K)
            cnt += 1
            if cnt > 500:
                break

        prec = precision(recommends, tests)
        rec = recall(recommends, tests)
        cov = coverage(recommends, all_items)

        self.log.info("precision: %.3f", prec)
        self.log.info("recall: %.3f", rec)
        self.log.info("coverage: %.3f", cov)

class Recommender(BaseAlgo):

    def __init__(self, *args, **kwargs):
        self(Recommender, self).__init__(*args, **kwargs)

    @abstractmethod
    def fit(self, data, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def recommend(self, user, n=None, candidates=None, ratings=None):
        raise NotImplementedError()
