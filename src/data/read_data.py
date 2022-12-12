"""data io"""

# load packages
import os
import numpy as np

# path
_PATH = "data/"


def load_data(name: str, read_labels: bool = True):
    """
    load data by folder name
    :param name: the name of the folder / dataset
    :param read_labels: true to read labels, false only read data
    :return numpy array
    """
    folder_path = os.path.join(_PATH, name)
    X = np.load(os.path.join(folder_path, f"{name}.npy"))
    if read_labels:
        y = np.load(os.path.join(folder_path, "labels.npy"))
        return X, y
    return X
