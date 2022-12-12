"""distortion metric"""

# load packages
import os
import torch

# constants
EPS = 1e-15

# pairwise distances
def _euclidean_pairwise_dist(data_mat):
    """
    compute the pairwise euclidean distance matrix
    || x - y || ^ 2 = || x || ^ 2 - 2 < x, y > + || y || ^ 2

    :param data_mat: of N by D
    :return dist_mat: of N by N
    """
    dist_mat = torch.cdist(
        data_mat, data_mat, p=2, compute_mode="donot_use_mm_for_euclid_dist"
    )
    return dist_mat


def _lorentz_pairwise_dist(data_mat):
    """lorentz distance"""
    dim = data_mat.size(1) - 1
    x0 = data_mat.narrow(-1, 0, 1)
    xr = data_mat.narrow(-1, 1, dim)
    inner_neg = x0 @ x0.T - xr @ xr.T
    dist_mat = torch.arccosh((inner_neg).clamp(min=1.0 + EPS))
    dist_mat.fill_diagonal_(0.0)
    return dist_mat


# distortion losses
def _scaled_pairwise_dist_loss(emb_dists_selected, real_dists_selected):
    """
    scaled version that is more presumably more compatible with optimizing distortion
    (d_e / mean(d_e) - d_r / mean(d_r)) ** 2
    """
    loss = torch.mean(
        (
            (emb_dists_selected / emb_dists_selected.mean())
            - (real_dists_selected / real_dists_selected.mean())
        )
        ** 2
    )
    return loss


def _distortion_loss(emb_dists_selected, real_dists_selected):
    """directly use average distortion as the loss"""
    loss = torch.mean(emb_dists_selected / (real_dists_selected)) * torch.mean(
        real_dists_selected / (emb_dists_selected)
    )
    return loss


def _select_upper_triangular(emb_dists, real_dists):
    """select the upper triangular portion of the distance matrix"""
    mask = torch.triu(torch.ones_like(real_dists), diagonal=1) > 0
    emb_dists_selected = torch.masked_select(emb_dists, mask)
    real_dists_selected = torch.masked_select(real_dists, mask)
    return emb_dists_selected, real_dists_selected


def distortion(X, X_latent, is_hyperbolic=True):
    """compute the distortion loss in training"""
    # compute pairwise distances
    real_dists = _euclidean_pairwise_dist(X)
    emb_dists = (
        _lorentz_pairwise_dist(X_latent)
        if is_hyperbolic
        else _euclidean_pairwise_dist(X_latent)
    )

    # select upper triangular portion
    emb_dists_selected, real_dists_selected = _select_upper_triangular(
        emb_dists, real_dists
    )

    # compute distortion loss
    distortion_loss = _distortion_loss(emb_dists_selected, real_dists_selected)
    scaled_loss = _scaled_pairwise_dist_loss(emb_dists_selected, real_dists_selected)
    return distortion_loss, scaled_loss
