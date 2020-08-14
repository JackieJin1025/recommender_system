import numpy as np
from numba import njit


def sparsity(arr):
    return 1 - (np.sum((arr != 0)) / (arr.shape[0] * arr.shape[1]))


def RMSE(pred, actual, ignore_na=True):
    """
        pred_rating: array
        actual_rating: array
    """
    pred = np.array(pred)
    actual = np.array(actual)
    if not ignore_na:
        m = np.nanmean(pred)
        pred = np.where(np.isnan(pred), m, pred)
    a = np.sqrt(np.nanmean((pred - actual) ** 2))

    return a


def MAE(pred, actual, ignore_na=True):
    pred = np.array(pred)
    actual = np.array(actual)
    if not ignore_na:
        m = np.nanmean(pred)
        pred = np.where(np.isnan(pred), m, pred)
    a = np.nanmean(np.abs(pred - actual))

    return a


def precision_recall_at_k(pred, actual, k=10, threshold=3.5):
    """Return precision and recall at k metrics for each user"""
    if actual.index.nunique() != len(actual):
        raise ValueError("index of actual is not unique")

    if pred.index.nunique() != len(pred):
        raise ValueError("index of pred is not unique")

    actual = actual[actual > threshold]
    pred = pred[pred > threshold]
    if k is not None:
        pred.sort_values(ascending=False, inplace=True)
        pred = pred[:k]

    # Number of relevant items
    n_rel = len(actual)
    # Number of recommended items in top k
    n_rec_k = len(pred)

    joined_index = pred.index.intersection(actual.index)
    n_rel_and_rec_k = len(joined_index)
    precision = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
    recall = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precision, recall


def precision(recommends, tests, threshold=3.5):
    """
    :param recommends: dict { userID : recommended items  }
    :param tests: dict { userID : true items  }
    :return: float
        Precision
    """
    n_union = 0.
    user_sum = 0.
    for user_id, items in recommends.items():
        recommend_set = set(items)
        tests = tests[user_id]
        tests = tests[tests > threshold]
        test_set = set(tests)
        n_union += len(recommend_set & test_set)
        user_sum += len(test_set)

    return n_union / user_sum


def recall(recommends, tests, threshold=3.5):
    """
        @param recommends:   { userID : recommended items  }
        @param tests:  test set: { userID : true items  }
        @return: Recall
    """
    n_union = 0.
    recommend_sum = 0.
    for user_id, items in recommends.items():
        recommend_set = set(items)
        tests = tests[user_id]
        tests = tests[tests > threshold]
        test_set = set(tests)
        n_union += len(recommend_set & test_set)
        recommend_sum += len(recommend_set)

    return n_union / recommend_sum


def coverage(recommends, all_items):
    """
        @param recommends : dict { userID : Items }
        @param all_items :  list/set items
    """
    recommend_items = set()
    for _, items in recommends.items():
        for item in items:
            recommend_items.add(item)
    return len(recommend_items) / len(all_items)


def popularity(item_popular, recommends):
    """
        @param item_popular:  dict { itemID : popularity}
        @param recommends :  dict { userID : Items }
        @return: average popularity
    """
    popularity = 0.
    n = 0.
    for _, items in recommends.items():
        for item in items:
            popularity += np.log(1. + item_popular.get(item, 0.))
            n += 1
    return popularity / n


def _evaluate(actual, pu, qi, bu=None, bi=None, global_mean=None):
    """
    :param actual: ratings m x n
    :param pu: user latent factor m x k
    :param qi: item latent factor n x k
    :param bu: user_offsets: m x 1
    :param bi: item_offsets: n x 1
    :param global_mean: int
    :return:
    """
    mask = actual != 0
    pred = pu.dot(qi.T)
    if bu is not None:
        pred += bu.reshape(-1, 1)
    if bi is not None:
        pred += bi.reshape(1, -1)
    if global_mean is not None:
        pred += global_mean
    pred, actual = pred[mask], actual[mask]
    mae = MAE(pred, actual)
    rmse = RMSE(pred, actual)
    return rmse, mae