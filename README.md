# Recommender_system

This is a generic recommender_system which provides various collaborative filterting algorithims including user-user, item-item, SVD, Factorization Machines, and so on. 

## Getting Started

Here is a simple example showing how you can use dataset movielens (1m), split it into test and train dataset, and compute the MAE and RMSE of FUNKSVD:

```python 
from recsys.cf.funksvd import FunkSVD
from recsys.utils.data import train_test_split, load_movielen_data
from recsys.utils.debug import LogUtil

LogUtil.configLog()
model = FunkSVD(learning_rate=0.001, reg=0.005, n_epochs=100, n_factors=30)
ratings, users, movies = load_movielen_data()

training, testing = train_test_split(ratings)
model.fit(training)
model.eval(testing)

```
## Output

```
2020-08-14 08:35:29,392 - FunkSVD - INFO - start evaluating with 99950 test samples ...
100%|██████████| 6040/6040 [00:25<00:00, 239.45it/s]
2020-08-14 08:35:55,025 - FunkSVD - INFO - rmse: 0.880, mae: 0.688
2020-08-14 08:35:55,026 - FunkSVD - INFO - evaluation takes 25.633
```

Another example showing recommender
```python 
from recsys.cf.recommender import TopN
from recsys.cf.selector import NotRatedSelector, RatedSelector
from recsys.cf.funksvd import FunkSVD
from recsys.utils.data import load_movielen_data
from recsys.utils.debug import LogUtil
import pandas as pd

pd.set_option('display.max_columns', None)
LogUtil.configLog()

ratings, users, movies = load_movielen_data()
movies.set_index('item', inplace=True)
users.set_index('user', inplace=True)

predictor = FunkSVD(learning_rate=0.001, reg=0.005, n_epochs=100, n_factors=30)
unrated_selector = NotRatedSelector()
model = TopN(predictor, unrated_selector)

model.fit(ratings)

user_id = 1
n = 10

model1 = TopN(predictor, unrated_selector)
print('Top %d recommended movies for user %s' %(n, user_id))
print(model.recommend(user_id, n, mapping=movies))
```

## Output

```
Top 10 recommended movies for user 1
         score                                   Title            Genres
item                                                                    
2905  4.845173                          Sanjuro (1962)  Action|Adventure
1949  4.730582           Man for All Seasons, A (1966)             Drama
318   4.671630        Shawshank Redemption, The (1994)             Drama
670   4.662056  World of Apu, The (Apur Sansar) (1959)             Drama
3338  4.661879                  For All Mankind (1989)       Documentary
326   4.653455                 To Live (Huozhe) (1994)             Drama
3022  4.649896                     General, The (1927)            Comedy
668   4.643326                  Pather Panchali (1955)             Drama
953   4.635666            It's a Wonderful Life (1946)             Drama
905   4.628309            It Happened One Night (1934)            Comedy

Process finished with exit code 0

```

## Dataset
MovieLens 1M movie ratings. Stable benchmark dataset. 1 million ratings from 6000 users on 4000 movies. Released 2/2003.
Further details can be found https://grouplens.org/datasets/movielens/1m/

## Authors

* **Jackie Jin** - *Initial work* - [JackieJin1025](https://github.com/JackieJin1025)


## Acknowledgments

* The implementation is inspired by various projects including https://github.com/lenskit/lkpy, https://surprise.readthedocs.io/en/stable/, and so on.
