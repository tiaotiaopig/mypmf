import matplotlib.pyplot as plt
import numpy as np
from LoadData import load_rating_data, spilt_rating_dat
from PMF import PMF

if __name__ == "__main__":
    file_path = "data/ml-100k/u.data"
    pmf = PMF(num_feat=10, epsilon=1, _lambda=0.1, momentum=0.9, max_epoch=12, num_batches=100, batch_size=1000)
    ratings = load_rating_data(file_path)
    print("user:%d item:%d feature:%d" % (len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])), pmf.num_feat))
    train, test = spilt_rating_dat(ratings, 0.2)
    pmf.fit(train, test)

    plt.plot(range(pmf.max_epoch), pmf.rmse_train, marker='o', label='Train Data')
    plt.plot(range(pmf.max_epoch), pmf.rmse_test, marker='v', label='Test Data')
    plt.title('The MovieLens Dataset Learning Curve')
    plt.xlabel('Number of Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()
