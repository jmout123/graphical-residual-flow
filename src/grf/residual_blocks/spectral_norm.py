"""
Spectral Normalization of weight matrices of fully-connected layers.

Adapted from:
Invertible Residual Networks http://proceedings.mlr.press/v97/behrmann19a/behrmann19a.pdf
"""

import torch

from torch.nn.functional import normalize


class SpectralNorm():

    _version = 1

    def __init__(self, coeff, n_power_iterations, epsilon):
        self.coeff = coeff
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.epsilon = epsilon
    

    def __call__(self, module, inputs):
        setattr(module, '_weight', self.compute_weight(module, power_iterate=module.training))


    def compute_weight(self, module, power_iterate):
        W = getattr(module, 'weight_original')
        W = torch.mul(module.mask, W)

        u = getattr(module, 'weight_u')
        v = getattr(module, 'weight_v')
        sigma_original_log = getattr(module, 'weight_original_sigma')
        sigma_log = getattr(module, 'weight_sigma')

        if power_iterate:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to u^T @ W @ v, where u 
                    # and v are the first left and right singular vectors.
                    # This power iteration produces approximations of u and v.
                    v = normalize(torch.mv(W.t(), u), dim=0, eps=self.epsilon, out=v)
                    u = normalize(torch.mv(W, v), dim=0, eps=self.epsilon, out=u)
                if self.n_power_iterations > 0:
                    u = u.clone()
                    v = v.clone()

        sigma = torch.dot(u, torch.mv(W, v))
        sigma_original_log.copy_(sigma.detach())
        factor = torch.max(torch.ones(1).to(W.device), sigma/self.coeff)
        sigma_log.copy_(torch.ones(1).to(W.device)*self.coeff)

        weight = W/factor

        return weight

    
    def _solve_v_and_rescale(self, W, u, target_sigma):
        # Tries to returns a vector `v` s.t. `u = normalize(W @ v)`
        # (the invariant at top of this class) and `u @ W @ v = sigma`.
        # This uses pinverse in case W^T W is not invertible.
        v = torch.chain_matmul(W.t().mm(W).pinverse(), W.t(), u.unsqueeze(1)).squeeze(1)
        return v.mul_(target_sigma/torch.dot(u, torch.mv(W, v)))


    @staticmethod
    def apply(module, c, n, epsilon):
        sn = SpectralNorm(c, n, epsilon)
        W = module._parameters['_weight']
        # Mask weight matrix
        W.data = torch.mul(module.mask, W)

        with torch.no_grad():
            h, w = W.size()
            # randomly initialize u and v
            u = normalize(W.new_empty(h).normal_(0,1), dim=0, eps=sn.epsilon)
            v = normalize(W.new_empty(w).normal_(0,1), dim=0, eps=sn.epsilon)
        
        # Rename 'weight' as 'weight_original'
        delattr(module, '_weight')
        module.register_parameter('weight_original', W)

        # We still need to assign weight back as 'weight' since other 
        # operations may assume that it exists. 
        setattr(module, '_weight', W.data)
        module.register_buffer('weight_u', u)
        module.register_buffer('weight_v', v)
        module.register_buffer('weight_original_sigma', torch.ones(1).to(W.device))
        module.register_buffer('weight_sigma', torch.ones(1).to(W.device))

        module.register_forward_pre_hook(sn)

        module._register_state_dict_hook(SpectralNormStateDictHook(sn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(sn))

        return sn


class SpectralNormLoadStateDictPreHook(object):

    def __init__(self, sn):
        self.sn = sn

    def __call__(self, state_dict, prefix, local_metadata, strict,
                 missing_keys, unexpected_keys, error_msgs):
        sn = self.sn
        version = local_metadata.get('spectral_norm', {}).get('weight.version', None)
        if version is None or version < 1:
            with torch.no_grad():
                weight_original = state_dict[prefix + 'weight_original']
                weight = state_dict.pop(prefix + 'weight')
                sigma = (weight_original / weight).mean()
                u = state_dict[prefix + 'weight_u']
                v = sn._solve_v_and_rescale(weight, u, sigma)
                state_dict[prefix + 'weight_v'] = v


class SpectralNormStateDictHook(object):

    def __init__(self, sn):
        self.sn = sn

    def __call__(self, module, state_dict, prefix, local_metadata):
        if 'spectral_norm' not in local_metadata:
            local_metadata['spectral_norm'] = {}
        key = 'weight.version'
        if key in local_metadata['spectral_norm']:
            raise RuntimeError("Unexpected key in metadata['spectral_norm']: {}".format(key))
        local_metadata['spectral_norm'][key] = self.sn._version

    
def spectral_norm(module, c, n, epsilon=1e-12):
    """Apply spectral normalization to weight matrix of NN layer.
            W = c*W/sigma(W)    if c/sigma(W) < 1
        where 
            sigma(W) = max_{a:a not 0} (||Wa||_2) / (||a||_2),
        in other words, sigma(W) is the spectral norm of W.
    
    Parameters
    ----------
        module : nn.Module
            Layer containing weight matrix to normalize
        c : float
            Scaling coefficient
        n : int
            Number of power iterations to compute spectral norm
        epsilon : float
            Epsilon for numerical stability in calculating the norms

    Returns
    -------
        The original module with the spectral norm hook
    """
    SpectralNorm.apply(module, c, n, epsilon)
    return module
