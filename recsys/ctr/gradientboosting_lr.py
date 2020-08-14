from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression

from recsys.utils.data import load_ctr_data


def _convert_one_hot(y_pred_train, num_leaf):
    """
    :param y_pred_train:  m x n where m is number of obs and n is is number of feature
    :param num_leaf: maximum number of leave nodes defined in the gradient boosting
    :return: m x (n_estimators * num_leaf)
    """
    m, n_estimators = y_pred_train.shape

    y_pred_train = y_pred_train.astype(int)
    transformed_training_matrix = np.zeros([m, n_estimators * num_leaf], dtype=np.int64)  # N * num_tress * num_leafs
    for i in range(0, m):
        idx = np.arange(n_estimators) * num_leaf + y_pred_train[i]
        transformed_training_matrix[i][idx] = 1
    return transformed_training_matrix


class GBDT_LR(BaseEstimator):

    """
        gradient boosting + logistic regression for ctr
    """
    def __init__(self, **kwargs):
        self.gbdt = GradientBoostingClassifier(**kwargs)
        self.lr = LogisticRegression(random_state=0)

    def fit(self, X_train, y_train):
        gbc = self.gbdt
        gbc.fit(X_train, y_train)

        num_leaf = gbc.max_leaf_nodes
        y_pred_train = gbc.apply(X_train)
        y_pred_train = y_pred_train.reshape(y_pred_train.shape[0], -1)
        y_pred_train = y_pred_train - y_pred_train.min(axis=0).reshape(1, -1)
        transformed_train = _convert_one_hot(y_pred_train, num_leaf)

        self.lr.fit(transformed_train, y_train)

    def _transform(self, X_test):
        gbc = self.gbdt
        num_leaf = gbc.max_leaf_nodes
        y_pred_test = gbc.apply(X_test)
        y_pred_test = y_pred_test.reshape(y_pred_test.shape[0], -1)
        y_pred_test = y_pred_test - y_pred_test.min(axis=0).reshape(1, -1)
        transformed_test = _convert_one_hot(y_pred_test, num_leaf)
        return transformed_test

    def predict(self, X_test):
        transformed_test = self._transform(X_test)
        lr_pred = self.lr.predict(transformed_test)
        return lr_pred

    def predict_prob(self, X_test):
        transformed_test = self._transform(X_test)
        pred_prob = self.lr.predict_prob(transformed_test)
        return pred_prob


if __name__ == '__main__':
    gbdt_lr = GBDT_LR(n_estimators=5, max_leaf_nodes=10)
    print(gbdt_lr.get_params())

    X_train, y_train = load_ctr_data()
    print(X_train.head())
    print(y_train.head())