import pickle
import random
from operator import itemgetter
from main.algorithm.basealgo import BaseAlgo
import numpy as np
from main.util.debug import Timer
from main.util.data import load_movielen_data


class LFM(BaseAlgo):
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
        m = len(self.users)
        n = len(self.items)
        # initilize latent factor matrix for users and items
        self.user_p = np.random.normal(size=(m,k))
        self.item_q = np.random.normal(size=(n, k))

    def select_negatives(self, user_items):
        """
            @param user_items: positive samples from user
            @return: return nagetive samples
        """
        samples = dict()
        items = list(self.items)
        for item in user_items:
            samples[item] = 1

        n_negative = 0
        n = len(user_items)
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
        epochs = self.epochs
        clock = Timer()
        rmat_array = self.rmat.toarray()

        for epoch in range(epochs):
            self.log.info("epoch {} started: ".format(epoch))
            for uid in range(rmat_array.shape[0]):
                user = self.users[uid]
                ratings = rmat_array[uid, :]
                user_items = self.items[np.argwhere(ratings != 0).flatten()]
                select_samples = self.select_negatives(user_items)
                for item, rui in select_samples.items():
                    err = rui - self.predict(user, item)
                    uid = self.users.get_loc(user)
                    iid = self.items.get_loc(item)
                    user_latent = self.user_p[uid, :]
                    movie_latent = self.item_q[iid, :]
                    
                    # gradient descent 
                    self.user_p[uid, :] += lr * (err * movie_latent - rr * user_latent)
                    self.item_q[iid, :] += lr * (err * user_latent - rr * movie_latent)
            e0 = clock.restart()
            loss = self.loss()
            e1 = clock.restart()
            self.log.info("loss: {}".format(loss))
            self.log.info("time elapsed: {}, {}".format(e0, e1))

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
        rmat_array = self.rmat.toarray()
        y = (rmat_array != 0).astype(int)
        pre = self.user_p.dot(self.item_q.T)
        yhat = 1.0 / (1 + np.exp(-pre))
        err = y - yhat
        c = np.sum(np.square(err)) + rr * np.sum(np.square(self.user_p)) + rr * np.sum(np.square(self.item_q))
        return c

    def predict(self, user, item):
        uid = self.users.get_loc(user)
        iid = self.items.get_loc(item)
        pre = np.dot(self.user_p[uid, :], self.item_q[iid, :])
        # convert to sigmoid
        return 1.0 / (1 + np.exp(-pre))


if __name__ == '__main__':

    ratings, users, movies = load_movielen_data()
    model = LFM(k=10)
    print(model.get_params())
    model.fit(ratings)