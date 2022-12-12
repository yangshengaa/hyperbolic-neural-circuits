"""useful layers"""

# load packages
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

# init params
INIT_MEAN = 0
INIT_STD = 0.001
EPS = 1e-15

# ============ genearl layers ============


def create_linear_layers(hidden_dim, num_layers, nl):
    """create linear layers of the same hidden dimensions"""
    # input part
    layer_list = []
    for _ in range(num_layers - 1):
        layer_list.append(nn.Linear(hidden_dim, hidden_dim))
        layer_list.append(nl)

    # final hidden output layer
    layer_list.append(nn.Linear(hidden_dim, hidden_dim))
    layer = nn.Sequential(*layer_list)
    return layer


def create_geodesic_layers(hidden_dim, num_layers, nl):
    """create geodesic layers of the same hidden dimensions"""
    # input part
    layer_list = []
    for _ in range(num_layers - 1):
        layer_list.append(GeodesicLorentz(hidden_dim, hidden_dim))
        layer_list.append(nl)
        layer_list.append(ExpLift())  # map back to Lorentz

    # final hidden output layer
    layer_list.append(GeodesicLorentz(hidden_dim, hidden_dim))
    layer = nn.Sequential(*layer_list)
    return layer


# ========== hyperbolic related layers ==============
def exp_lift(X: torch.Tensor) -> torch.Tensor:
    """a exponential map from euclidean to Lorentz"""
    # print(X)
    x_norm = X.norm(dim=1, keepdim=True)
    x0 = torch.cosh(x_norm)
    xr = torch.sinh(x_norm) * X / (x_norm + EPS)
    mapped_x = torch.cat([x0, xr], dim=-1)
    # print(mapped_x)
    # print(mapped_x.shape)
    return mapped_x


class ExpLift(nn.Module):
    """a module wrapper for the lifting function"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, X):
        return exp_lift(X)


class GeodesicLorentz(nn.Module):
    """lorentz geodesic layer: mapping lorentz back to Euclidean"""

    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # init
        self.z = Parameter(torch.normal(INIT_MEAN, INIT_STD, (in_dim, out_dim)))
        self.a = Parameter(torch.normal(INIT_MEAN, INIT_STD, (1, out_dim)))

    # aux
    def get_w(self, z, a):
        """get tangent space vectors from euclidean ones, by a exponential map (PP trick)"""
        z_norm = z.norm(dim=0, keepdim=True)
        w0 = z_norm * torch.sinh(a)
        wr = torch.cosh(a) * z
        w = torch.cat([w0, wr], dim=0)
        return w

    # distance
    def lorentz_dist2plane(self, W, X):
        """
        vectorized lorentz dist2plane

        arcsinh(-<w, x>_L / (sqrt(<w, w>_L)) * ||w||_L
        adapted from https://proceedings.mlr.press/v89/cho19a.html, and a thorough discussiong with Zhengchao
        """
        numerator = -X.narrow(-1, 0, 1) @ W[[0]] + X.narrow(-1, 1, self.in_dim) @ W[1:]
        z_norm = self.z.norm(dim=0, keepdim=True)
        denom = z_norm
        distance = torch.arcsinh(numerator / denom) * z_norm
        return distance

    def forward(self, X):
        W = self.get_w(self.z, self.a)
        euclidean_features = self.lorentz_dist2plane(W, X)
        # print(euclidean_features)
        return euclidean_features
