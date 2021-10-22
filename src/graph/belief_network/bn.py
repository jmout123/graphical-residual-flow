class BeliefNetwork():
    """Interface for belief networks with latent variables z
    and observed variable x. The full joint can subsequently be 
    factorized as p(x,z) = p(x|z)p(z), where p(x|z) and p(z) are 
    the likelihood and prior terms, respectively.
    """

    def __init__(self):
        super(BeliefNetwork, self).__init__()

    
    def sample(self):
        """Draw samples from the forward graph g."""
        pass


    def log_likelihood(self, x, z):
        """Compute the log-likelihood log(p(x|z))"""
        pass


    def log_prior(self, z):
        """Compute the log-prior log(p(z))"""
        pass


    def log_joint(self, x, z):
        """Compute the log-joint log(p(x,z))"""
        pass


    def get_num_latent(self):
        pass


    def get_num_obs(self):
        pass


    def topological_order(self):
        pass


    def inv_topological_order(self):
        pass