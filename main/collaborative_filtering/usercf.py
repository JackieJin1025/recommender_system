#! /usr/bin/python3
# coding=utf-8
'''
'''
from collections import defaultdict
from operator import itemgetter
import pandas as pd
import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from main.collaborative_filtering.basecf import BaseCF
from main.collaborative_filtering.itemcf import ItemCF
from main.util.debug import Timer


class UserCF(BaseCF):

    def __init__(self, discount_popularity=False, *args, **kwargs):
        self.discount_popularity = discount_popularity
        self.user_sim_matrix = None
        super(UserCF, self).__init__(*args, **kwargs)

    def _train(self):
        """
            train model
        """
        self.user_sim_matrix = self.user_similarity()

    def _save(self):
        c = Timer()
        # cache trained parameter
        with open(self.filename, 'wb') as outfile:
            pickle.dump(self.user_sim_matrix, outfile)
        e = c.restart()

        self.log.info("saving user_sim_matrix to %s takes %.3f", self.filename, e)

    def _load(self):
        c = Timer()
        with open(self.filename, 'rb') as infile:
            self.user_sim_matrix = pickle.load(infile)
        e = c.restart()
        self.log.info("loading user_sim_matrix from %s takes %.3f", self.filename, e)

    def user_similarity(self):
        """
            define user similarity based on cosine of movies watched
            train: dict user -> a list of items
        """
        train = self.user2items
        item2idx = self.item2idx
        user2idx = self.user2idx
        n = len(item2idx)
        m = len(user2idx)

        arr = np.zeros([m, n])
        for user, items in train.items():
            user_idx = user2idx[user]
            for item in items:

                discount_factor = 1
                if self.discount_popularity:
                    # discount popular item in the similarity weights
                    f = len(self.item2users.get(item))
                    discount_factor = np.log(1+ f)
                arr[user_idx, item2idx[item]] = 1.0 / discount_factor

        M = cosine_similarity(arr)
        user_sim = defaultdict(dict)
        for u1, idx1 in user2idx.items():
            for u2, idx2 in user2idx.items():
                if u1 == u2:
                    continue
                user_sim[u1][u2] = M[idx1][idx2]

        return user_sim


    def recommend(self, user, N, K):
        """
            @param user:
            @param N:    number of recommended items
            @param K:    number of similiar users
            @return: (item, score)
        """


        related_items = self.user2items.get(user, set)
        recommends = defaultdict(int)
        similar_users = self.user_sim_matrix.get(user)
        for u, sim in sorted(similar_users.items(), key=itemgetter(1), reverse=True)[:K]:

            for item in self.user2items[u]:
                # if the item already belongs to user, skip
                if item in related_items:
                    continue
                # for a given item, add up similarities from all k users
                recommends[item] += sim
        res = dict(sorted(recommends.items(), key=itemgetter(1), reverse=True)[: N])
        self.log.info("start recommend %s with %s items via %s similar users" % (user, len(res), K))
        return res


    def recommend_users(self, users, N, K):
        """
            @param users:    a list of users
            @param N:    number of recommended items
            @param K:    number of similiar users
            @return:  {user : list of recommended items}
        """
        recommends = dict()
        for user in users:
            user_recommends = list(self.recommend(user, N, K).keys())
            recommends[user] = user_recommends

        return recommends





if __name__ == '__main__':
    base_dir = "/Users/Jackie/Work/RecommendationSystem/data/ml-1m"
    filename = "ratings.dat"
    movies = get_data(base_dir, "movies.dat", 'MovieID::Title::Genres'.split("::"))
    ratings = get_data(base_dir,"ratings.dat", "UserID::MovieID::Rating::Timestamp".split("::"))
    users = get_data(base_dir,"users.dat", "UserID::Gender::Age::Occupation::Zip-code".split("::"))
    user_ids = list(users['UserID'].unique())
    train_data = ratings[['UserID', 'MovieID', 'Rating']]

    train_data, test_data = train_test_split(ratings[['UserID', 'MovieID', 'Rating']], frac=0.2)

    #model = UserCF(discount_popularity=True)
    model = ItemCF()
    model.train(train_data)
    model.test(10, test_data)