"""measure hyperbolicity"""

# load packages
import os
import argparse

import numpy as np

# load file
from data import load_data, create_gaussian_synthetic
from util import hyperbolicity_sample

# =========== arguments ===========
parser = argparse.ArgumentParser()

# data
parser.add_argument("--data", type=str, default="mnist", help="the name of the dataset")
parser.add_argument(
    "--num-samples",
    type=int,
    default=50000,
    help="number of sampels to approximate hyperbolicity",
)
parser.add_argument(
    "--n", type=int, default=500, help="the number of samples in synthetic"
)
parser.add_argument("--p", type=int, default=20, help="the dimension to generate")
args = parser.parse_args()

# ======== main driver ========
def main():
    # load data
    print("reading data")
    if args.data == "syn":
        X = create_gaussian_synthetic(args.n, args.p)
    else:
        X = load_data(args.data, read_labels=False)

    # measure hyperbolicity
    delta = hyperbolicity_sample(X, num_samples=args.num_samples)

    # log data
    with open("result/hyperbolicity.txt", "a+") as f:
        data_info = f'syn_{args.n}_{args.p}' if args.data == 'syn' else args.data
        f.write(f"{data_info},{delta}\n")


if __name__ == "__main__":
    main()
