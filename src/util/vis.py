"""
visualization of the Lorentz features in L^2
"""

# load pakcages
import os 
import torch 
import torch.nn as nn 

# =========== stereographic projection ==========
def lorentz_to_poincare(X: torch.Tensor) -> torch.Tensor:
    """ map lorentz to poincare """
    d = X.shape[1] - 1
    x0 = X.narrow(-1, 0, 1)
    xr = X.narrow(-1, 1, d)
    x_projected = xr / (1 + x0)
    return x_projected

