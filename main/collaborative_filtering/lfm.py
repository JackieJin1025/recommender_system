import pickle
import random
from operator import itemgetter
from main.collaborative_filtering.basecf import BaseCF
import numpy as np
from main.util.debug import Timer


class LFM(BaseCF):
    """
        latent factor model
    """
    def __init__(self, k, learning_rate=0.01, regularization_rate=0.1, epochs=10, *args, **kwargs):
        """
        :param k: dimension of latent factor
        :param learning_rate:
        :param regularization_rate:
        :param epochs:
        :param args:
        :param kwargs:
        """
        self.k = k
        self.regularization_rate = regularization_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        # latent factor
        self.user_p = None # m x k
        self.item_q = None # n x k
        super(LFM, self).__init__(*args, **kwargs)


    def init_latent_factor(self):
        k = self.k
        m = len(self.user2idx)
        n = len(self.item2idx)
        # initilize latent factor matrix for users and items
        self.user_p = np.random.normal(size = (m ,k))
        self.item_q = np.random.normal(size=  (n, k))


    def select_negatives(self, user_movies):
        """
            @param user_movies: positive samples from user
            @return: return nagetive samples
        """
        samples = dict()
        items = list(self.item2idx.keys())
        for item in user_movies:
            samples[item] = 1

        n_negative = 0
        n = len(user_movies)
        for _ in range(n* 5):
            # have max iter in case num(user_movies) is too large
            negitive_sample = random.choice(items)
            if negitive_sample in samples:
                continue
            samples[negitive_sample] = 0
            n_negative += 1
            if n_negative > n:
                break
        return samples


    def _train(self):
        self.init_latent_factor()
        lr = self.learning_rate
        rr = self.regularization_rate
        epochs =  self.epochs
        clock = Timer()
        for epoch in range(epochs):
            print("epoch {} started: ".format(epoch))
            for user, user_movies in self.user2items.items():
                select_samples = self.select_negatives(user_movies)
                for item, rui in select_samples.items():
                    err = rui - self.predict(user, item)
                    uid = self.user2idx[user]
                    iid = self.item2idx[item]
                    user_latent = self.user_p[uid, :]
                    movie_latent = self.item_q[iid, :]
                    
                    # gradient descent 
                    self.user_p[uid, :] += lr * (err * movie_latent - rr * user_latent)
                    #print(self.user_p[user])
                    self.item_q[iid, :] += lr * (err * user_latent - rr * movie_latent)
            e0 = clock.restart()
            loss = self.loss()
            e1 = clock.restart()
            print("loss: {}".format(loss))
            print("time elapsed: {}, {}".format(e0, e1))

    def _save(self):
        # cache trained parameter
        with open(self.filename, 'wb') as outfile:
            pickle.dump((self.user_p, self.item_q), outfile)
        self.log.info("saved user_p, item_q to %s", self.filename)

    def _load(self):
        with open(self.filename, 'rb') as infile:
            self.user_p, self.item_q = pickle.load(infile)

        self.log.info("loaded user_p, item_q from %s", self.filename)

    def loss(self):
        """loss function """
        rr = self.regularization_rate
        m = len(self.user2idx)
        n = len(self.item2idx)

        y = np.zeros( (m, n))
        for user, uid in self.user2idx.items():
            for item, iid in self.item2idx.items():
                if (user, item) in self.ratings:
                    y[uid, iid] = 1

        pre = self.user_p.dot(self.item_q.T)
        yhat = 1.0 / (1 + np.exp(-pre))
        err = y - yhat
        c = np.sum(np.square(err))  +  rr * np.sum(np.square(self.user_p)) +  rr * np.sum(np.square(self.item_q))
        return c


    def predict(self, user, item):
        uid = self.user2idx[user]
        iid = self.item2idx[item]
        pre =  np.dot(self.user_p[uid, :], self.item_q[iid, :])
        # convert to sigmoid
        return 1.0 / (1 + np.exp(-pre))


    def recommend_user(self, user, N, K):
        """
            recommend top N items based on top k similar items from each item the user likes
        """
        print("start recommend %s with %s items" % (user, N))
        seen_items = self.user2items[user]

        recommends = dict()
        for item, _ in self.item2idx.items():
            if item in seen_items:
                continue
            recommends[item] = self.predict(user, item)
        return dict(sorted(recommends.items(), key=itemgetter(1), reverse=True)[: N])

