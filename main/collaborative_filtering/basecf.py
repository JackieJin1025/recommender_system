from collections import defaultdict
from main.util.debug import LogUtil
from main.util.metric import precision, recall, coverage
from os import path



class BaseCF(object):
    def __init__(self, filename=None):
        """
            :param filename: if provided, cache parameters trained
        """
        self.filename = filename

        self.user2items = defaultdict(set)
        self.item2users = defaultdict(set)
        self.item2idx = dict()
        self.idx2item = dict()
        self.user2idx = dict()
        self.idx2user = dict()
        self.ratings = dict()
        self.item_popularity = defaultdict(int)
        self.log = LogUtil.getLogger(self.__class__.__name__)

    def init_data(self, origin_data):
        """
        :param origin_data: dataframe 'UserID', 'MovieID', 'Rating'
        :return:
        """
        items = set()
        users = set()

        for user, item, rating in origin_data.values:
            self.user2items[user].add(item)
            self.item2users[item].add(user)
            self.ratings[(user, item)] = rating
            self.item_popularity[item] += 1
            items.add(item)
            users.add(user)

        for idx, e in enumerate(items):
            self.item2idx[e] = idx
            self.idx2item[idx] = e

        for idx, e in enumerate(users):
            self.user2idx[e] = idx
            self.idx2user[idx] = e

    def train(self, origin_data):
        self.init_data(origin_data)
        flag = self.filename is not None

        if flag and path.exists(self.filename):
            self._load()
            return

        self.log.info('start training ...')
        self._train()
        self.log.info('training is finished ...')
        if flag:
            self._save()


    def _load(self):
        raise Exception("not implemented")

    def _save(self):
        raise Exception("not implemented")

    def recommend(self, user_id, n, K):
        raise Exception("not implemented")

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
            cnt+= 1
            if cnt > 500:
                break

        prec = precision(recommends, tests)
        rec = recall(recommends, tests)
        cov = coverage(recommends, all_items)

        self.log.info("precision: %.3f", prec)
        self.log.info("recall: %.3f", rec)
        self.log.info("coverage: %.3f", cov)



