from lenskit.algorithms.user_knn import UserUser
from main.algorithm.usercf import UserCF
from main.utils.data import load_movielen_data
from main.utils.debug import Timer, LogUtil

LogUtil.configLog()
ratings, users, movies = load_movielen_data()
model0 = UserCF(min_threshold=0.1, min_nn=5, max_nn=20)
model0.fit(ratings)

model = UserUser(nnbrs=20, min_nbrs=5, min_sim=0.1, center=False)
model.fit(ratings)

user = 1
movies = list(movies.item.astype(int))
movies = [1]
clock = Timer()
for _ in range(5):
    df = model.predict_for_user(user, movies)
    print(clock.restart())

print("="*60)

for _ in range(5):
    df0 = model0.predict_for_user(user, movies)
    print(clock.restart())


print(df.describe())
print(df0.describe())
