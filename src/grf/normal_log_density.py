import torch
import torch.nn as nn

from math import pi


class NormalLogDensity(nn.Module):
    def __init__(self):
        super(NormalLogDensity, self).__init__()
        self.register_buffer("pi", torch.tensor(pi))

    
    def forward(self, x, loc, scales):
        return -0.5*(torch.log(2*self.pi) + ((x - loc)/scales)**2) - torch.log(scales)


class StandardNormalLogDensity(nn.Module):
    def __init__(self):
        super(StandardNormalLogDensity, self).__init__()
        self.register_buffer("pi", torch.tensor(pi))

        
    def forward(self, x):
        return (-0.5*(torch.log(self.pi*2) + x**2)).sum(1)
