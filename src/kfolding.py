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

    # Get all images and labels
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

    # Save all images and labels to a pkl file
    with open('../full_dataset.pkl', 'wb') as f:
        pickle.dump(splits, f)

    # Setting up K-fold
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    # Split the dataset into folds
    splits2 = list(kfold.split(images))

    # Save all folds to a pkl file for future use
    with open('../kfold_dataset.pkl', 'wb') as f:
        pickle.dump(splits2, f)

