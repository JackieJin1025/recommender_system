import sys
from recommender.utils.data import load_movielen_data, train_test_split
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

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
X_train = X_train.todense()