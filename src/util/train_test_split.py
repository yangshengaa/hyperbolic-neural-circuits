"""train validation test split"""

# load packages
import os
import numpy as np


def train_val_test_split(n, seed=2022, test_size=0.25, val_size=0.15):
    """k-fold validation and testing"""
    np.random.seed(seed)

    index_arr = np.arange(n)
    np.random.shuffle(index_arr)
    test_index = index_arr[-int(n * test_size) :]
    train_val_index = index_arr[: -int(n * test_size)]
    val_index = train_val_index[-int(n * val_size) :]
    train_index = train_val_index[: -int(n * val_size)]
    # folds_index = np.split(train_val_index, fold)

    # # concat train val
    # fold_index_list = []
    # for i in range(fold):
    #     fold_train = np.hstack((folds_index[j] for j in range(fold) if j != i))
    #     fold_val = folds_index[i]
    #     fold_index_list.append((fold_train, fold_val))

    # return fold_index_list, test_index
    return train_index, val_index, test_index
