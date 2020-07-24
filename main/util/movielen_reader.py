from main.util.data import get_data
import os
from pathlib import Path


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
    load_movielen_data()