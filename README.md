# Recommender_system

This is a generic recommender_system which provides various collaborative filterting algorithims including user-user, item-item, SVD, Factorization Machines, and so on. 

## Getting Started

Here is a simple example showing how you can use dataset movielens (1m), split it into test and train dataset, and compute the MAE and RMSE of FUNKSVD:

```python 
from recommender.algorithm.funksvd import FunkSVD
from recommender.utils.data import train_test_split, load_movielen_data
from recommender.utils.debug import LogUtil

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

Another example showing recommender
```python 
from recommender.algorithm.dummy import DummyPredictor
from recommender.algorithm.recommender import TopN
from recommender.algorithm.selector import NotRatedSelector, RatedSelector
from recommender.algorithm.funksvd import FunkSVD
from recommender.utils.data import load_movielen_data
from recommender.utils.debug import LogUtil
import pandas as pd

pd.set_option('display.max_columns', None)
LogUtil.configLog()

ratings, users, movies = load_movielen_data()
movies.set_index('item', inplace=True)
users.set_index('user', inplace=True)

predictor = FunkSVD(learning_rate=0.001, reg=0.005, n_epochs=100, n_factors=30)
unrated_selector = NotRatedSelector()
model1 = TopN(predictor, unrated_selector)

model1.fit(ratings)

user_id = 1
n = 10
print('Top %d recommended movies for user %s' %(n, user_id))
print(model1.recommend(user_id, n, mapping=movies))
```

## Output

```
Top 10 recommended movies for user 1
         score                                              Title  
item                                                                
2905  4.857698                                     Sanjuro (1962)   
318   4.768027                   Shawshank Redemption, The (1994)   
904   4.704973                                 Rear Window (1954)   
3338  4.695507                             For All Mankind (1989)   
670   4.690026             World of Apu, The (Apur Sansar) (1959)   
2019  4.686721  Seven Samurai (The Magnificent Seven) (Shichin...   
858   4.677496                              Godfather, The (1972)   
3470  4.670209                                 Dersu Uzala (1974)   
923   4.664409                                Citizen Kane (1941)   
326   4.663084                            To Live (Huozhe) (1994)   

```

## Dataset
MovieLens 1M movie ratings. Stable benchmark dataset. 1 million ratings from 6000 users on 4000 movies. Released 2/2003.
Further details can be found https://grouplens.org/datasets/movielens/1m/

## Authors

* **Jackie Jin** - *Initial work* - [JackieJin1025](https://github.com/JackieJin1025)


## Acknowledgments

* The implementation is inspired by various projects including https://github.com/lenskit/lkpy, https://surprise.readthedocs.io/en/stable/, and so on.
