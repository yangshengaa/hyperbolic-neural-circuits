"""measure hyperbolicity of a dataset (approximate)"""

# load packages
import os
import time
import numpy as np
from tqdm import tqdm


def hyperbolicity_sample(X: np.ndarray, num_samples: int = 50000) -> float:
    """
    compute the approximate relative hyperbolicity of a dataset given its pairwise distance
    adapted from https://github.com/HazyResearch/hgcn/blob/master/utils/hyperbolicity.py

    :param X: the input matrix
    :param num_samples: the number of samples to perform approximation
    :return the approximated hyperbolicity
    """
    n = X.shape[0]
    curr_time = time.time()
    max_hyps = -float("inf")
    diameter = -float("inf")
    for _ in tqdm(range(num_samples)):
        curr_time = time.time()
        node_tuple = np.random.choice(
            range(n), 4, replace=False
        )  # randomly sampled four points
        s = []
        try:
            sample_0 = X[node_tuple[0]]
            sample_1 = X[node_tuple[1]]
            sample_2 = X[node_tuple[2]]
            sample_3 = X[node_tuple[3]]

            # euclidean l2 distances
            d01 = np.linalg.norm(sample_0 - sample_1)
            d23 = np.linalg.norm(sample_2 - sample_3)
            d02 = np.linalg.norm(sample_0 - sample_2)
            d13 = np.linalg.norm(sample_1 - sample_3)
            d03 = np.linalg.norm(sample_0 - sample_3)
            d12 = np.linalg.norm(sample_1 - sample_2)
            s.append(d01 + d23)
            s.append(d02 + d13)
            s.append(d03 + d12)
            s.sort()
            cur_hyp = (s[-1] - s[-2]) / 2
            max_hyps = max(cur_hyp, max_hyps)
            diameter = max([diameter, d01, d02, d03, d12, d13, d23])
        except Exception as e:
            continue
    print("Time for hyp: ", time.time() - curr_time)
    relative_hyp = 2 * max_hyps / diameter
    return relative_hyp
