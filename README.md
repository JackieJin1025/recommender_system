# Recommender_system

This is a a generic recommender_system which provides various collaborative filterting algorithims including user-user, item-item, SVD, Factorization Machines, and so on. 

## Getting Started

Here is a simple example showing how you can use dataset movielens (1m), split it into test and train dataset, and compute the MAE and RMSE of FUNKSVD:


```python 
from main.algorithm.funksvd import FunkSVD
from main.utils.data import train_test_split, load_movielen_data
from main.utils.debug import LogUtil

LogUtil.configLog()
model = FunkSVD(learning_rate=0.001, reg=0.005, n_epochs=100, n_factors=30)
ratings, users, movies = load_movielen_data()

training, testing = train_test_split(ratings)
model.fit(training)
model.eval(testing)

```
## Output

```
2020-08-08 19:47:21,277 - FunkSVD - INFO - start evaluating with 99950 test samples ...
100%|██████████| 6040/6040 [00:33<00:00, 179.58it/s]
2020-08-08 19:47:55,289 - FunkSVD - INFO - rmse: 0.879, mae: 0.689
2020-08-08 19:47:55,289 - FunkSVD - INFO - evaluation takes 34.011
```

## Dataset
MovieLens 1M movie ratings. Stable benchmark dataset. 1 million ratings from 6000 users on 4000 movies. Released 2/2003.
Further details can be found https://grouplens.org/datasets/movielens/1m/

## Authors

* **Jackie Jin** - *Initial work* - [JackieJin1025](https://github.com/JackieJin1025)


## Acknowledgments

* The implementation is inspired by various projects including https://github.com/lenskit/lkpy, https://surprise.readthedocs.io/en/stable/, and so on.
