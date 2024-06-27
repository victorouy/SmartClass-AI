# This file is to setup the kfold data. It will create two pkl files. One pkl file will be the entire dataset. The
# other pkl file will be the indices for the folds.

import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def fetchData():
    path = '../dataset-cleaned/'
    allaimges = []
    clsLabel = []

    for idx, cls in enumerate(['angry', 'engaged', 'happy', 'neutral']):
        Cpath = os.path.join(path, cls)
        F = os.listdir(Cpath)
        for im in F:
            allaimges.append(os.path.join(Cpath, im))
            clsLabel.append(idx)

    return allaimges, clsLabel


if __name__ == '__main__':
    images, labels = fetchData()

    splits = {
        'images': images,
        'labels': labels,
    }

    with open('../full_dataset.pkl', 'wb') as f:
        pickle.dump(splits, f)


