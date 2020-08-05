from sklearn.feature_extraction import DictVectorizer
import tensorflow.compat.v1 as tf
from main.utils.data import load_movielen_data, train_test_split
from main.algorithm.basic import Predictor
import numpy as np

from main.utils.debug import LogUtil

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
        super(TensorFM, self).__init__(*args, **kwargs)

    def fit(self, train):
        v = DictVectorizer(sparse=False)
        train_data = [dict(user_id=str(row[0]), item_id=str(row[1])) for row in train[['user', 'item']].values]
        x_train = v.fit_transform(train_data)
        y_train = train['rating'].astype(np.float).values.reshape(-1, 1)
        x_train = x_train[:2000, :]
        y_train = y_train[:2000, :]
        n, p = x_train.shape
        # number of latent factors
        k = self.n_factors
        # design matrix
        X = tf.placeholder(tf.float32, shape=[n, p])
        y = tf.placeholder(tf.float32, shape=[n, 1])
        # bias and weights
        w0 = tf.Variable(tf.zeros([1]))
        W = tf.Variable(tf.zeros([p]))
        # interaction factors, randomly initialized
        V = tf.Variable(tf.random_normal([k, p], stddev=0.01))  # V
        # estimate of y, initialized to 0.
        # y_hat = tf.Variable(tf.zeros([n, 1]))

        linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(W, X), 1, keepdims=True))
        term1 = tf.pow(tf.matmul(X, tf.transpose(V)), 2)
        term2 = tf.matmul(tf.pow(X, 2), tf.pow(tf.transpose(V), 2))
        interactions = tf.reduce_sum(tf.subtract(term1, term2), 1, keepdims=True)

        y_hat = tf.add(linear_terms, interactions)
        # L2 regularized sum of squares loss function over W and V
        lambda_w = tf.constant(self.reg, name='lambda_w')
        lambda_v = tf.constant(self.reg, name='lambda_v')
        l2_norm = tf.add(tf.reduce_sum(tf.multiply(lambda_w, tf.pow(W, 2))), tf.reduce_sum(tf.multiply(lambda_v, tf.pow(V, 2))))

        error = tf.reduce_mean(tf.squared_difference(y, y_hat))
        loss = tf.add(error, l2_norm)

        # eta = tf.constant(0.1)
        optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr).minimize(loss)

        # Launch the graph.
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(self.n_epochs):
                indices = np.arange(n)
                np.random.shuffle(indices)
                x_data, y_data = x_train[indices], y_train[indices]
                sess.run(optimizer, feed_dict={X: x_data, y: y_data})

                if epoch % 10 == 0:
                    mse = sess.run(error, feed_dict={X: x_data, y: y_data})
                    total_loss = sess.run(loss, feed_dict={X: x_data, y: y_data})
                    self.log.info('epoch %d, RMSE: %.3f, Loss (regularized error): %.3f', epoch, np.sqrt(mse), total_loss)

            # self.log.info('Predictions: %s', sess.run(y_hat, feed_dict={X: x_data, y: y_data}))
            self.log.info('Learnt weights: %s', sess.run(W, feed_dict={X: x_data, y: y_data}))
            self.log.info('Learnt factors: %s', sess.run(V, feed_dict={X: x_data, y: y_data}))


if __name__ == '__main__':
    LogUtil.configLog()
    ratings, users, items = load_movielen_data()
    train, test = train_test_split(ratings, frac=0.1)
    model = TensorFM(n_epochs=300, n_factors=5, learning_rate=0.01)
    model.fit(train)