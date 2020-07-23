import os
import pandas as pd

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

