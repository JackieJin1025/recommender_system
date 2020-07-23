from collections import defaultdict
from operator import itemgetter
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from main.collaborative_filtering.basecf import BaseCF


class ItemCF(BaseCF):

    def __init__(self, discount_popularity=False, *args, **kwargs):
        self.discount_popularity = discount_popularity
        self.item_sim_matrix = None
        super(ItemCF, self).__init__(*args, **kwargs)

    def _train(self):
        self.item_sim_matrix = self.item_similarity()


    def _save(self):
        # cache trained parameter
        with open(self.filename, 'wb') as outfile:
            pickle.dump(self.item_sim_matrix, outfile)
        self.log.info("save item_sim_matrix to %s", self.filename)


    def _load(self):
        with open(self.filename, 'rb') as infile:
            self.item_sim_matrix =  pickle.load(infile)
        self.log.info("loaded item_sim_matrix from %s", self.filename)

    def item_similarity(self):
        """
            define item similarity based on cosine of users who watched the item
            train: dict user -> a list of items
        """
        item2users = self.item2users
        user2items = self.user2items
        item2idx = self.item2idx
        user2idx = self.user2idx
        n = len(item2idx)
        m = len(user2idx)

        arr = np.zeros([n, m])
        for item, users in item2users.items():
            item_idx = item2idx[item]
            for user in users:
                discount_factor = 1
                if self.discount_popularity:
                    discount_factor = np.log(1+ len(user2items[user]))
                arr[item_idx, user2idx[user]] = 1.0 / discount_factor

        M = cosine_similarity(arr)
        item_sim = defaultdict(dict)
        for i1, idx1 in item2idx.items():
            for i2, idx2 in item2idx.items():
                if i1 == i2:
                    continue
                item_sim[i1][i2] = M[idx1][idx2]

        return item_sim


    def recommend(self, user, N, K):
        """
            recommend top N items based on top k similar items from each item the user likes
        """

        related_items = self.user2items.get(user, set)
        recommends = defaultdict(int)
        for item in related_items:
            sim_map = self.item_sim_matrix.get(item)
            for e, sim in  sorted(sim_map.items(), key=itemgetter(1), reverse=True)[: K]:
                if e == item:
                    continue
                recommends[e] += sim
        res = dict(sorted(recommends.items(), key=itemgetter(1), reverse=True)[: N])

        self.log.info("recommend %s with %s items via %s similar items", user, len(res), K)
        return res



    def recommend_users(self, users, N, K):
        """
            @param users:    a list of users
            @return:  {user : list of recommended items}
        """
        recommends = dict()
        for user in users:
            user_recommends = list(self.recommend(user, N, K).keys())
            recommends[user] = user_recommends

        return recommends