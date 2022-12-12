"""hyperbolic autoencoder"""

# load packages
import torch
import torch.nn as nn

# load file
from .layers import GeodesicLorentz, create_linear_layers, exp_lift

# constants
EPS = 1e-15


class AE(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_hidden_layers,
        latent_dim,
        nl=nn.ReLU(),
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.latent_dim = latent_dim
        self.nl = nl

        # construct encoder
        e_input_layer = nn.Linear(in_dim, hidden_dim)
        e_hidden_layer = create_linear_layers(hidden_dim, num_hidden_layers, nl)
        e_output_layer = nn.Linear(hidden_dim, latent_dim)
        self.encoder = nn.Sequential(
            e_input_layer, nl, e_hidden_layer, nl, e_output_layer
        )

        # construct decoder
        d_input_layer = nn.Linear(latent_dim, hidden_dim)
        d_hidden_layer = create_linear_layers(hidden_dim, num_hidden_layers, nl)
        d_output_layer = nn.Linear(hidden_dim, in_dim)
        self.decoder = nn.Sequential(
            d_input_layer, nl, d_hidden_layer, nl, d_output_layer
        )

    def forward(self, X):
        """encoder + decoder"""
        X_latent = self.encoder(X)
        X_recon = self.decoder(X_latent)
        return X_latent, X_recon


class HAE(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_hidden_layers,
        latent_dim,
        nl=nn.ReLU(),
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.latent_dim = latent_dim
        self.nl = nl

        # construct encoder
        e_input_layer = nn.Linear(in_dim, hidden_dim)
        e_hidden_layer = create_linear_layers(hidden_dim, num_hidden_layers, nl)
        e_output_layer = nn.Linear(hidden_dim, latent_dim)
        self.encoder = nn.Sequential(
            e_input_layer, nl, e_hidden_layer, nl, e_output_layer
        )

        # construct decoder
        d_input_layer = GeodesicLorentz(latent_dim, hidden_dim)
        d_hidden_layer = create_linear_layers(hidden_dim, num_hidden_layers, nl)
        d_output_layer = nn.Linear(hidden_dim, in_dim)
        self.decoder = nn.Sequential(
            d_input_layer, nl, d_hidden_layer, nl, d_output_layer
        )

    def forward(self, X):
        """hyperbolic autoencoder"""
        X_latent_euc = self.encoder(X)
        X_latent_hyp = exp_lift(X_latent_euc)
        X_recon = self.decoder(X)
        return X_latent_hyp, X_recon
