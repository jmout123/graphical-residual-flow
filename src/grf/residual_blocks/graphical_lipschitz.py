import torch.nn as nn
import torch

from math import ceil, log
from warnings import warn
from torch.nn.parameter import Parameter

from grf.residual_blocks.spectral_norm import spectral_norm

#### Activation Functions ####
class LipSwish(nn.Module):

    def __init__(self, beta=None):
        super(LipSwish, self).__init__()
        if beta is None:
            self.beta = Parameter(torch.tensor(1.0))
        else :
            self.beta = beta
    
    def forward(self, x):
        return x*torch.sigmoid(self.beta*x)/1.1

    def prime(self, x):
        return self.beta*self(x) + \
            torch.sigmoid(self.beta*x)*(1/1.1 - self.beta*self(x))


def elu(x):
    n = x < 0
    p = x >= 0
    return (p*x) + (torch.exp((n*x)) - 1)

def elu_prime(x):
    n = x < 0
    p = x >= 0
    return (torch.exp((n*x)))


def tanh_prime(x):
    # tanh'(x) = sech^2(x) = 1 - tanh^2(x)
    return 1 - (torch.tanh(x))**2


def sigmoid_prime(x):
    return torch.sigmoid(x)*(1 - torch.sigmoid(x))


class MaskedLayer(nn.Module):

    def __init__(self, in_size, out_size, mask):
        super(MaskedLayer, self).__init__()
        self.mask = mask
        self.layer = nn.Linear(in_size, out_size).double()
        self._weight = self.layer.weight
        self._bias = self.layer.bias

    
    def forward(self, x):
        W = self._weight
        x = torch.addmm(self._bias, x, W.transpose(0,1))
        return x


class MaskedSpecNormMLP(nn.Module):
    def __init__(self, in_size, hidden, out_size, masks, activation_function, 
                coeff, n_power_iterations):
        super(MaskedSpecNormMLP, self).__init__()
        self.masks = masks
        self.num_hidden = len(hidden)
        self.coeff = coeff
        self.n_power_iterations = n_power_iterations

        in_size = in_size
        l1 = [in_size] + hidden
        l2 = hidden + [out_size]

        layers = []
        for m, l in enumerate(zip(l1, l2)):
            layers.append(self._spectral_norm(MaskedLayer(l[0], l[1], masks[m]).to(masks[0].device)))
        self.layers = nn.ModuleList(layers)
        self.activation_func = activation_function

    
    def forward(self, x):
        layer_output = []
        for l in range(len(self.layers)-1):
            x = self.layers[l](x)
            layer_output.append(x)
            x = self.activation_func(x)
        x = self.layers[-1](x)
        return x, layer_output

    
    def _spectral_norm(self, layer):
        return spectral_norm(layer, self.coeff, self.n_power_iterations)

    
    def lipschitz(self):
        # Lip(g) = prod_i Lip(W_i @ x + b_i)
        #   where Lip(W_i @ x + b_i) = ||W_i||
        # NOTE: Lip(activation function) = 1
        L = 1
        for l in self.layers:
            L *= l.weight_sigma.item()
        return L


    def largest_singular_values(self):
        sigmas = []
        for l in self.layers:
            W = l._weight
            _, S, _ = torch.svd(W, compute_uv=False)
            sigmas.append(S[0].item())
        return sigmas


class GraphicalLipschitzResidualBlock(nn.Module):
    """ f(z) = z + g(z|x)
            where the weight matrices of g(z|x) are normalized st Lip(g) < 1.
    """

    def __init__(self, in_dim, hidden_dims, masks, cond_dim=0, coeff=0.97,
                 n_power_iterations=5, activation_function='lipswish'):
        super(GraphicalLipschitzResidualBlock, self).__init__()
        self.in_dim = in_dim + cond_dim
        self.out_dim = in_dim

        if len(hidden_dims) == 0:
            raise Exception("Residual block requires at least one hidden layer")
        self.hidden_dims = hidden_dims

        if activation_function == 'elu': 
            act_func = elu
            self.act_func_prime = elu_prime
        elif activation_function == 'tanh': 
            act_func = torch.tanh
            self.act_func_prime = tanh_prime
        elif activation_function == 'sigmoid': 
            act_func = torch.sigmoid
            self.act_func_prime = sigmoid_prime
        elif activation_function == 'lipswish':
            act_func = LipSwish()
            self.act_func_prime = act_func.prime

        self.g = MaskedSpecNormMLP(self.in_dim, hidden_dims, self.out_dim, 
                    masks, act_func, coeff, n_power_iterations)

    
    def forward(self, eps, x=None):
        if x is not None: input = torch.cat((eps, x), dim=1)
        else: input = eps
        g, layer_outputs = self.g(input)
        j = self.jacobian_det(layer_outputs)
        return eps + g, j


    def jacobian_det(self, layer_outputs):
        # log det J_f = trace(log(J_f))
        diag_J = torch.abs(
            self._jacobian_det(layer_outputs))
        return torch.log(diag_J).sum(1)


    def _jacobian_det(self, layer_outputs):
        L = self.g.num_hidden
        N = layer_outputs[0].shape[0]

        W0 = self.g.layers[0]._weight[:,:self.out_dim]
        J = W0.expand(N,-1,-1)
        for l in range(1,L+1):
            W = self.g.layers[l]._weight
            h_prime = self.act_func_prime(layer_outputs[l-1]).unsqueeze(dim=1)
            J = h_prime*J.transpose(1,2)
            J = J.reshape(N*self.out_dim, self.hidden_dims[l-1]).transpose(0,1)
            J = torch.mm(W,J)
            J = torch.stack(J.chunk(N, dim=1), dim=0)       

        # log det J_f = trace(log(J_f))
        diag_J = torch.stack([torch.diag(j) for j in torch.unbind(J, dim=0)], dim=0)

        # J_f(eps|x) = I + J_g(eps|x)
        diag_J = diag_J + 1
        return diag_J

    def inverse(self, z, x, maxT, epsilon):
        # Compute y st f(y|x) = z using newton-like fixed-point iteration
        y = z
        e = torch.norm(y)
        for _ in range(maxT):
            if e <= epsilon:
                break
            if x is not None: input = torch.cat((y, x), dim=1)
            else: input = y
            g, layer_outputs = self.g(input)
            diag_J = self._jacobian_det(layer_outputs)

            f = y + g
            y = y - (f - z)/diag_J
            
            e = torch.norm(f - z)

        if x is not None: input = torch.cat((y, x), dim=1)
        else: input = y
        g, layer_outputs = self.g(input)
        j = self.jacobian_det(layer_outputs)
        return y, j