from collections import defaultdict
from operator import itemgetter
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from main.collaborative_filtering.basecf import BaseCF
from main.util.data import get_data
import os
import pandas as pd


class ItemCF(BaseCF):

    def __init__(self, discount_popularity=False, 
                 min_nn=1, 
                 max_nn=50,
                 min_threshold=None,
                 *args, **kwargs):
        self.discount_popularity = discount_popularity        
        self.min_nn = min_nn 
        self.max_nn = max_nn 
        self.min_threshold = min_threshold

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
        # user by item
        rmat = self.rmat
        #normalize by item std
        item_norm = np.linalg.norm(rmat.toarray(), axis=0)
        nmat = rmat.toarray() / item_norm.reshape(1, -1)
        item_sim = nmat.T.dot(nmat)
        return item_sim

    def predict_for_user(self, user, items, ratings=None):
        min_threshold = self.min_threshold
        min_nn = self.min_nn
        max_nn = self.max_nn

        result = dict()
        # convert rmat to array
        rmat_array = self.rmat.toarray()

        items = np.array(items)
        valid_mask = self.items.get_indexer(items) >= 0

        if np.sum(~valid_mask) > 0:
            self.log.warning("%s are not valid" % items[~valid_mask])

        items = items[valid_mask]
        for item in items:

            upos = self.users.get_loc(user)
            ipos = self.items.get_loc(item)

            # idx with decending similarities with itself
            item_idx = np.argsort(self.item_sim_matrix[ipos, :])[::-1][1:]

            # sim need to meet min_threshold
            if min_threshold is not None:
                item_idx = item_idx[self.item_sim_matrix[ipos, item_idx] > min_threshold]

            # sim need to meet min_threshold
            item_idx = item_idx[rmat_array[upos, item_idx] != 0]

            item_idx = item_idx[:max_nn]
            if len(item_idx) < min_nn:
                self.log.warning("item %s does not have enough neighbors (%s < %s)", item, len(item_idx), min_nn)
                continue

            ratings = rmat_array[upos, item_idx]
            sim_wt = self.item_sim_matrix[ipos, item_idx]
            rating = ratings.dot(sim_wt) / sim_wt.sum()
            print(item)
            result[item] = rating
        return result


    def item_similarity_deprecated(self):
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
            to be deprecated
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
            to be deprecated
            @param users:    a list of users
            @return:  {user : list of recommended items}
        """
        recommends = dict()
        for user in users:
            user_recommends = list(self.recommend(user, N, K).keys())
            recommends[user] = user_recommends

        return recommends


if __name__ == '__main__':
    base_dir = "/Users/Jackie/Work/RecommendationSystem/data/ml-1m"
    movies = get_data(os.path.join(base_dir, "movies.dat"), 'MovieID::Title::Genres'.split("::"))
    ratings = get_data(os.path.join(base_dir, "ratings.dat") , "UserID::MovieID::Rating::Timestamp".split("::"))
    ratings = ratings.rename(columns={'UserID': 'user', 'MovieID': 'item', 'Rating': 'rating'})
    ratings[['user', 'item']] = ratings[['user', 'item']].astype(int)
    ratings['rating'] = ratings['rating'].astype(float)
    itemcf = ItemCF()
    itemcf.fit(ratings)

    user = 1
    movies = list(movies.MovieID.astype(int))
    res = itemcf.predict_for_user(user, movies)
    df = pd.Series(res)
    print(df)