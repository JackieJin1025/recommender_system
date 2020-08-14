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