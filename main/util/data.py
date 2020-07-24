import os
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from main.util.debug import LogUtil, Timer


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


def train_test_split(ratings, frac=0.2 , group='UserID', seed=1):
    """
        split data into train and test by frac
        if group is provide, split date into train and test by frac in each group
    """
    log = LogUtil.getLogger('train_test_split')
    log.info("start splitting test and train data takes")
    clock = Timer()
    if group:
        ratings_test = pd.DataFrame()
        for k, v in ratings.groupby(group):
            ratings_test = ratings_test.append(v.sample(frac=frac, random_state=seed))
    else:
        ratings_test = ratings.sample(frac=frac, random_sate=seed)

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

