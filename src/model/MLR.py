"""hyperbolic multilogistic regression"""

# load packages
import torch
import torch.nn as nn

# load file
from .layers import (
    GeodesicLorentz,
    ExpLift,
    create_linear_layers,
    create_geodesic_layers,
)


class MLR(nn.Module):
    """multilogistic regression"""

    def __init__(
        self, in_dim, hidden_dim, num_hidden_layers, out_dim, nl=nn.ReLU()
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.out_dim = out_dim
        self.nl = nl

        # construct model
        input_layer = nn.Linear(in_dim, hidden_dim)
        intermediate_layers = create_linear_layers(hidden_dim, num_hidden_layers, nl)
        out_layer = nn.Linear(hidden_dim, out_dim)
        self.clf = nn.Sequential(input_layer, nl, intermediate_layers, nl, out_layer)

        # softmax
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, X):
        """forward pass"""
        raw = self.clf(X)
        logits = self.softmax(raw)
        return logits


class HMLR(nn.Module):
    """hyperbolic multilogistic regression"""

    def __init__(
        self, in_dim, hidden_dim, num_hidden_layers, out_dim, nl=nn.ReLU()
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.out_dim = out_dim
        self.nl = nl

        # construct model
        input_layer = GeodesicLorentz(in_dim, hidden_dim)
        intermediate_layers = create_geodesic_layers(hidden_dim, num_hidden_layers, nl)
        out_layer = GeodesicLorentz(hidden_dim, out_dim)
        self.clf = nn.Sequential(
            ExpLift(),
            input_layer,
            nl,
            ExpLift(),
            intermediate_layers,
            nl,
            ExpLift(),
            out_layer,
        )

        # softmax
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, X):
        """forward pass"""
        raw = self.clf(X)
        logits = self.softmax(raw)
        return logits
