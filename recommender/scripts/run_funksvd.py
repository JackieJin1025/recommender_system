from recommender.algorithm.funksvd import FunkSVD
from recommender.utils.data import train_test_split, load_movielen_data
from recommender.utils.debug import LogUtil

LogUtil.configLog()
model = FunkSVD(learning_rate=0.001, reg=0.005, n_epochs=100, n_factors=30)
ratings, users, movies = load_movielen_data()

training, testing = train_test_split(ratings)
model.fit(training)
model.eval(testing)