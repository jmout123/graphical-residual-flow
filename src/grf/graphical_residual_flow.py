import torch.distributions as dist
import torch.nn as nn
import torch

class GraphicalResidualFlow(nn.Module):

    def __init__(self, blocks, bn, eps_log_density, device,
                    generative=True):
        super(GraphicalResidualFlow, self).__init__()
        self.blocks = nn.ModuleList()
        for block in blocks:
            self.blocks.append(block)
        
        self.bn = bn
        self.latent_dim = bn.get_num_latent()

        self.device = device
        self.eps_log_density = eps_log_density

        # Generating Flow: z0 -> zT
        # Normalizing Flow: zT -> z0
        self.generative = generative

    
    def forward(self, *args, **kwargs):
        if self.generative:
            return self._forward_gen(*args, **kwargs)
        else:
            return self._forward_norm(*args, **kwargs)

    
    def _forward_gen(self, cond=None, z0=None, batch_size=100):
        j_total = 0.0

        if cond is not None:
            batch_size = cond.shape[0]

        # Sample from reference distribution
        if z0 is None:
            z = dist.Normal(
                loc=torch.zeros(self.latent_dim, dtype=torch.float64),
                scale=torch.ones(self.latent_dim, dtype=torch.float64)
            ).sample((batch_size,)).to(self.device)
        else:
            z = z0
        z0 = z.detach().clone()

        # Forward pass through flow: z_{t+1} = z_t + g_t(z_t)
        for block in self.blocks:
            z, j = block(z, cond)
            j_total += j

        return z0, z, j_total

    
    def _forward_norm(self, z, cond=None):
        j_total = 0.0

        z0 = z
        for block in self.blocks:
            z0, j = block(z0, cond)
            j_total += j

        return z0, j_total


    def inverse(self, z, cond=None, maxT=10, epsilon=1e-5):
        j_total = 0.0

        with torch.no_grad():
            for block in reversed(self.blocks):
                z, j = block.inverse(z, cond, maxT, epsilon)
                j_total += j

        return z, j_total


    def shifted_reverse_kl(self, x, z, z0, j):
        """Minimize the negative variational evidence lower bound:
           E_{z~q}[log(q(z|x)) - log(p(x,z))]"""

        log_p_x_z = self.bn.log_joint(x, z)
        log_q_z_given_x = self.eps_log_density(z0) - j
        return (log_q_z_given_x - log_p_x_z).mean()

    
    def ll(self, z0, j):
        """Applies change-of-variable formula to determine the density of 
        a sample either generated by the flow (generating direction: z0 -> zT)
        or provided to the flow (normalizing direction: zT -> z0)
        """
        if self.generative:
            return (self.eps_log_density(z0) - j).mean()
        else:
            return (self.eps_log_density(z0) + j).mean()