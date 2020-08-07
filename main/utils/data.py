import os
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from main.utils.debug import LogUtil, Timer, timer


def get_data(filename, columns, delimiter ='::'):
    """
    :param filename: path of data source
    :param columns: column name for each column
    :param delimiter: delimiter to split a line
    :return: dataframe
    """
    log = LogUtil.getLogger('get_data')
    clock = Timer()
    with open(os.path.join(filename) , 'r') as infile:
        data = infile.readlines()
        df = pd.DataFrame([row.rstrip().split(delimiter) for row in data], columns=columns)

    e0 = clock.restart()
    log.info("loading data from %s with columns %s takes %.3f secs  ", filename, columns, e0)
    return df


def train_test_split(ratings, frac=0.1, group='user', seed=1):
    """
        split data into train and test by frac
        if group is provide, split date into train and test by frac in each group
    """
    log = LogUtil.getLogger('train_test_split')
    log.info("start splitting test and train data ...")
    clock = Timer()
    if group:
        ratings_test = ratings.groupby(group).apply(lambda x: x.sample(frac=frac, random_state=seed))
        ratings_test.index = ratings_test.index.droplevel(group)
    else:
        ratings_test = ratings.sample(frac=frac, random_state=seed)

    ratings_train = pd.merge(ratings, ratings_test, indicator=True, how='outer').query('_merge=="left_only"').drop(
        '_merge', axis=1)

    e0 = clock.restart()
    log.info("splitting test and train data takes %.3f secs", e0)
    return ratings_train, ratings_test


def sparse_ratings(ratings, users=None, items=None):
    """
    Convert a rating table to a sparse matrix of ratings.

    Args:
        ratings(pandas.DataFrame): a data table of (user, item, rating) triples.
        users(pandas.Index): an index of user IDs.
        items(pandas.Index): an index of items IDs.

    Returns:
            tuple containing the sparse matrix, user index, and item index.
    """
    if users is None:
        users = pd.Index(np.unique(ratings.user), name='user')
    if items is None:
        items = pd.Index(np.unique(ratings.item), name='item')

    row_ind = users.get_indexer(ratings.user).astype(np.intc)
    if np.any(row_ind < 0):
        raise ValueError('provided user index does not cover all users')
    col_ind = items.get_indexer(ratings.item).astype(np.intc)
    if np.any(col_ind < 0):
        raise ValueError('provided item index does not cover all users')

    if 'rating' in ratings.columns:
        vals = np.require(ratings.rating.values, np.float64)
    else:
        vals = None

    matrix = coo_matrix((vals, (row_ind, col_ind)), shape= (len(users), len(items)))
    return matrix, users, items


def batcher(X, y=None, w=None, batch_size=-1):
    """Split data to mini-batches.

    Parameters
    ----------
    X : {numpy.array, scipy.sparse.csr_matrix}, shape (n_samples, n_features)
        Training vector, where n_samples in the number of samples and
        n_features is the number of features.

    y : np.array or None, shape (n_samples,)
        Target vector relative to X.

    w : np.array or None, shape (n_samples,)
        Vector of sample weights.

    batch_size : int
        Size of batches.
        Use -1 for full-size batches

    Yields
    -------
    ret_x : {numpy.array, scipy.sparse.csr_matrix}, shape (batch_size, n_features)
        Same type as input

    ret_y : np.array or None, shape (batch_size,)

    ret_w : np.array or None, shape (batch_size,)
    """
    n_samples = X.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X[i:upper_bound]
        ret_y = None
        ret_w = None
        if y is not None:
            ret_y = y[i:i + batch_size]
        if w is not None:
            ret_w = w[i:i + batch_size]
        yield ret_x, ret_y, ret_w


def load_movielen_data():
    """
    :return: a tuple with three dataframes: ratings, users, movies
    """
    p = Path(__file__).parents[2]
    base_dir = os.path.join(p, 'data', 'ml-1m')
    movies = get_data(os.path.join(base_dir, "movies.dat"), 'MovieID::Title::Genres'.split("::"))
    ratings = get_data(os.path.join(base_dir, "ratings.dat"), "UserID::MovieID::Rating::Timestamp".split("::"))
    users = get_data(os.path.join(base_dir, "users.dat"), "UserID::Gender::Age::Occupation::Zip-code".split("::"))
    movies = movies.rename(columns={'MovieID': 'item'})
    movies['item'] = movies['item'].astype(int)
    users = users.rename(columns={'UserID': 'user'})
    users['user'] = users['user'].astype(int)
    ratings = ratings.rename(columns={'UserID': 'user', 'MovieID': 'item', 'Rating': 'rating'})
    ratings[['user', 'item']] = ratings[['user', 'item']].astype(int)
    ratings['rating'] = ratings['rating'].astype(float)
    return ratings, users, movies


if __name__ == '__main__':

    ratings, users, movies = load_movielen_data()
    print(ratings.head())
    print(ratings.describe())
    print(users.describe())

    train, test = train_test_split(ratings, frac=0.1, group='user')
    print(train.describe())
    print(test.describe())