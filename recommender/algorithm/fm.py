from sklearn.feature_extraction import DictVectorizer
import tensorflow.compat.v1 as tf
from recommender.utils.data import load_movielen_data, train_test_split, batcher
from recommender.algorithm.basic import Predictor
import numpy as np

from recommender.utils.debug import LogUtil

tf.compat.v1.disable_eager_execution()


class TensorFM(Predictor):
    """
    """
    def __init__(self, n_epochs, n_factors, learning_rate, reg=0.01, shuffle=False, *args, **kwargs):
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.reg = reg
        self.lr = learning_rate
        self.shuffle = shuffle
        self.dict_vectorizer = DictVectorizer(sparse=False)
        super(TensorFM, self).__init__(*args, **kwargs)

    def _convert_data(self, data):
        v = self.dict_vectorizer
        train_data = [dict(user_id=str(row[0]), item_id=str(row[1])) for row in data[['user', 'item']].values]
        if hasattr(v, 'feature_names_'):
            x_train = v.transform(train_data)
        else:
            x_train = v.fit_transform(train_data)

        y_train = None
        if 'rating' in data.columns:
            y_train = data['rating'].astype(np.float).values.reshape(-1, 1)
        return x_train, y_train

    def init_params(self, n_feature):
        """
        :param n_feature: the number of features in  x
        :return:
        """
        k = self.n_factors
        reg = self.reg
        lr = self.lr

        p = n_feature
        # design matrix
        X = tf.placeholder(tf.float32, shape=(None, p), name='X')
        y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
        # bias and weights
        W0 = tf.Variable(tf.zeros([1]))
        W = tf.Variable(tf.zeros([p]))
        # interaction factors, randomly initialized
        V = tf.Variable(tf.random_normal([k, p], stddev=0.01))  # V

        linear_terms = tf.add(W0, tf.reduce_sum(tf.multiply(W, X), 1, keepdims=True))
        term1 = tf.pow(tf.matmul(X, tf.transpose(V)), 2)
        term2 = tf.matmul(tf.pow(X, 2), tf.pow(tf.transpose(V), 2))
        interactions = tf.reduce_sum(tf.subtract(term1, term2), 1, keepdims=True)
        y_hat = tf.add(linear_terms, interactions, name='y_hat')
        # L2 regularized sum of squares loss function over W and V
        lambda_w = tf.constant(reg, name='lambda_w')
        lambda_v = tf.constant(reg, name='lambda_v')
        l2_norm = tf.add(tf.reduce_sum(tf.multiply(lambda_w, tf.pow(W, 2))), tf.reduce_sum(tf.multiply(lambda_v, tf.pow(V, 2))))

        error = tf.reduce_mean(tf.squared_difference(y, y_hat), name='error')
        loss = tf.add(error, l2_norm, name='loss')

        # eta = tf.constant(0.1)
        optimizer = tf.train.AdagradOptimizer(learning_rate=lr).minimize(loss)

        self.X = X
        self.y = y
        self.W = W
        self.W0 = W0
        self.V = V
        self.y_hat = y_hat
        self.error = error
        self.loss = loss
        self.optimizer = optimizer
        self.session = tf.Session()
        return X, y, W0, W, V, y_hat, error, loss, optimizer

    def fit(self, train):
        x_train, y_train = self._convert_data(train)
        # x_train = x_train[:2000, :]
        # y_train = y_train[:2000, :]
        n, p = x_train.shape
        X, y, W0, W, V, y_hat, error, loss, optimizer = self.init_params(p)

        # Launch the graph.
        init = tf.global_variables_initializer()
        sess = self.session
        sess.run(init)

        for epoch in range(self.n_epochs):
            # indices = np.arange(n)
            # np.random.shuffle(indices)
            # x_data, y_data = x_train[indices], y_train[indices]

            for x_data, y_data, _ in batcher(x_train, y_train, batch_size=1000):

                sess.run(optimizer, feed_dict={X: x_data, y: y_data})

            if epoch % 10 == 0:
                mse = sess.run(error, feed_dict={X: x_data, y: y_data})
                total_loss = sess.run(loss, feed_dict={X: x_data, y: y_data})
                self.log.info('epoch %d, RMSE: %.3f, Loss (regularized error): %.3f', epoch, np.sqrt(mse), total_loss)

        self.log.info('Learnt weights: %s', sess.run(W, feed_dict={X: x_data, y: y_data}))
        self.log.debug('Predictions: %s', sess.run(y_hat, feed_dict={X: x_data, y: y_data}))
        self.log.debug('Learnt factors: %s', sess.run(V, feed_dict={X: x_data, y: y_data}))

        return self

    def predict(self, data):
        X = self.X
        y = self.y
        y_hat = self.y_hat
        x_test, y_test = self._convert_data(data)
        sess = self.session
        pred = sess.run(y_hat, feed_dict={X: x_test, y: y_test})
        return pred

    def predict_for_user(self, user, items=None, ratings=None):
        pass



if __name__ == '__main__':
    LogUtil.configLog()
    ratings, users, items = load_movielen_data()
    train, test = train_test_split(ratings, frac=0.1)
    model = TensorFM(n_epochs=30, n_factors=5, learning_rate=0.01)
    model.fit(train)
    print(model.predict(test))