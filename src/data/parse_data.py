"""
scripts to parse data
"""

# load packages
import os
import pickle
import numpy as np
import pandas as pd


_DATA_PATH = "data/raw"

# parse methods
# ========== odor ==========
def standardize_and_select(df):
    """standardize each column and select ones with high variability"""
    df = df.select_dtypes(include=[float])  # select numeric columns
    col_std = df.std()
    df_selected = df.iloc[:, np.where(col_std > 0.5)[0]].astype(float)
    print(f"selected {df_selected.shape[1]}/{df.shape[1]} columns")

    # standardization
    df_selected_standardized = (df_selected - df_selected.mean()) / df_selected.std(
        ddof=0
    )
    return df_selected_standardized


def parse_odor():
    """parse the odor dataset"""
    # tomato
    tomato_df = pd.read_excel(
        os.path.join(_DATA_PATH, "odor/curbio_9537_mmc2.xlsx"),
        sheet_name="clean",
        header=None,
    )
    tomato_df_selected = standardize_and_select(tomato_df)
    tomato_np = tomato_df_selected.to_numpy()

    folder_path = os.path.join("data/tomato")
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    np.save(os.path.join(folder_path, "tomato.npy"), tomato_np)

    # strawberry
    strawberry_df = pd.read_excel(
        os.path.join(_DATA_PATH, "odor/pone.0088446.s005.xlsx"),
        sheet_name="clean",
        header=None,
    )
    strawberry_df_selected = standardize_and_select(strawberry_df)
    strawberry_np = strawberry_df_selected.to_numpy()

    folder_path = os.path.join("data/strawberry")
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    np.save(os.path.join(folder_path, "strawberry.npy"), strawberry_np)

    # blueberry
    blueberry_df = pd.read_excel(
        os.path.join(_DATA_PATH, "odor/pone.0138494.s002.xlsx"),
        sheet_name="clean",
        header=None,
    )
    blueberry_df_selected = standardize_and_select(blueberry_df)
    blueberry_np = blueberry_df_selected.to_numpy()

    folder_path = os.path.join("data/blueberry")
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    np.save(os.path.join(folder_path, "blueberry.npy"), blueberry_np)


# ======== mnist =========
def parse_mnist():
    """parse mnist dataset"""
    train_df = pd.read_csv(os.path.join(_DATA_PATH, "mnist", "mnist_train.csv"))
    train_X = train_df.iloc[:, 1:].to_numpy()
    train_y = train_df.iloc[:, 0].to_numpy()

    test_df = pd.read_csv(os.path.join(_DATA_PATH, "mnist", "mnist_test.csv"))
    test_X = test_df.iloc[:, 1:].to_numpy()
    test_y = test_df.iloc[:, 0].to_numpy()

    X = np.vstack((train_X, test_X))
    y = np.hstack((train_y, test_y))

    # standardize
    # X_selected = X[:, X.std(axis=0) > 1]
    # X_selected = (X_selected - X_selected.mean(axis=0, keepdims=True)) / X_selected.std(
    #     axis=0, keepdims=True
    # )
    X_selected = X / 256
    print(X_selected.shape)

    # save
    folder_path = "data/mnist"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    np.save(os.path.join(folder_path, "mnist.npy"), X_selected)
    np.save(os.path.join(folder_path, "labels.npy"), y)


# ======== cifar 10 =========
def unpickle(file):
    with open(file, "rb") as fo:
        cur_dict = pickle.load(fo, encoding="bytes")
    return cur_dict


def to_grayscale(X):
    """convert cifar 10 to grayscale"""
    X = X.astype(float)
    X_r = X[:, :1024]
    X_g = X[:, 1024 : 1024 * 2]
    X_b = X[:, 1024 * 2 :]

    X_grayscale = 299 / 1000 * X_r + X_g * 587 / 1000 + X_b * 114 / 1000
    return X_grayscale


def parse_cifar10():
    """parse cifar10 dataset"""
    # load
    batch_1 = unpickle(os.path.join(_DATA_PATH, "cifar-10-batches-py", "data_batch_1"))
    batch_2 = unpickle(os.path.join(_DATA_PATH, "cifar-10-batches-py", "data_batch_2"))
    batch_3 = unpickle(os.path.join(_DATA_PATH, "cifar-10-batches-py", "data_batch_3"))
    batch_4 = unpickle(os.path.join(_DATA_PATH, "cifar-10-batches-py", "data_batch_4"))
    batch_5 = unpickle(os.path.join(_DATA_PATH, "cifar-10-batches-py", "data_batch_5"))

    gray_scale_X = [
        to_grayscale(batch[b"data"])
        for batch in (batch_1, batch_2, batch_3, batch_4, batch_5)
    ]
    X = np.vstack(gray_scale_X)
    labels = np.hstack(
        (batch[b"labels"] for batch in (batch_1, batch_2, batch_3, batch_4, batch_5))
    )

    # standardize X
    X = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True) / 10

    # save
    folder_path = "data/cifar10"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    np.save(os.path.join(folder_path, "cifar10.npy"), X)
    np.save(os.path.join(folder_path, "labels.npy"), labels)


if __name__ == "__main__":
    parse_odor()
    parse_cifar10()
    parse_mnist()
