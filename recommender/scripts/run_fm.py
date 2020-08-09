from recommender.utils.data import load_movielen_data, train_test_split
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import tensorflow.compat.v1 as tf


top = 1000
ratings, users, items = load_movielen_data()
train, test = train_test_split(ratings, frac=0.1)


train_data = [dict(user_id=str(row[0]), item_id=str(row[1])) for row in train[['user', 'item']].values]
y_train = train['rating'].astype(np.float).values.reshape(-1, 1)

test_data = [dict(user_id=str(row[0]), item_id=str(row[1])) for row in test[['user', 'item']].values]
y_test = test['rating'].astype(np.float).values.reshape(-1, 1)

v = DictVectorizer()
X_train = v.fit_transform(train_data)
X_test = v.transform(test_data)

X_train = X_train[:1000, :]
y_train = y_train[:1000, :]


n, p = X_train.shape
# number of latent factors
k = 5
# design matrix
X = tf.placeholder(tf.float32, shape=[n, p])
# target vector
y = tf.placeholder(tf.float32, shape=[n, 1])
# bias and weights
w0 = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.zeros([p]))
# interaction factors, randomly initialized
V = tf.Variable(tf.random_normal([k, p], stddev=0.01))  # V
# estimate of y, initialized to 0.
y_hat = tf.Variable(tf.zeros([n, 1]))


linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(W, X), 1, keepdims=True) )
term1 = tf.pow(tf.matmul(X, tf.transpose(V)), 2)
term2 = tf.matmul(tf.pow(X, 2), tf.pow(tf.transpose(V), 2))
interactions = tf.reduce_sum(tf.subtract(term1, term2), 1, keepdims=True)

# L2 regularized sum of squares loss function over W and V
lambda_w = tf.constant(0.001, name='lambda_w')
lambda_v = tf.constant(0.001, name='lambda_v')
l2_norm = tf.add(tf.reduce_sum(tf.multiply(lambda_w, tf.pow(W, 2))), tf.reduce_sum(tf.multiply(lambda_v, tf.pow(V, 2))))


error = tf.reduce_mean(tf.squared_difference(y, y_hat))
loss = tf.add(error, l2_norm)
# eta = tf.constant(0.1)
optimizer = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)


# that's a lot of iterations
N_EPOCHS = 1000
# Launch the graph.
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(N_EPOCHS):
        indices = np.arange(n)
        np.random.shuffle(indices)
        x_data_, y_data = X_train[indices], y_train[indices]
        sess.run(optimizer, feed_dict={X: X_train, y: y_train})

    print('MSE: ', sess.run(error, feed_dict={X: X_train, y: y_train}))
    print('Loss (regularized error):', sess.run(loss, feed_dict={X: X_train, y: y_train}))
    print('Predictions:', sess.run(y_hat, feed_dict={X: X_train, y: y_train}))
    print('Learnt weights:', sess.run(W, feed_dict={X: X_train, y: y_train}))
    print('Learnt factors:', sess.run(V, feed_dict={X: X_train, y: y_train}))