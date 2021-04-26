import numpy as np
import random


def load_rating_data(file_path='ml-100k/u.data'):
    """
    load movie lens 100k ratings from original rating file.
    need to download and put rating data in /data folder first.
    Source: http://www.grouplens.org/
    """
    prefer = []
    with open(file_path, 'r') as lines:
        for line in lines.readlines():
            (userid, movie_id, rating, ts) = line.split('\t')
            uid = int(userid)
            mid = int(movie_id)
            rat = float(rating)
            prefer.append([uid, mid, rat])
    return np.asarray(prefer)


def spilt_rating_dat(data, size=0.2):
    train_data = []
    test_data = []
    for line in data:
        rand = random.random()
        if rand < size:
            test_data.append(line)
        else:
            train_data.append(line)
    train_data = np.asarray(train_data)
    test_data = np.asarray(test_data)
    return train_data, test_data
