# Modification from: Faithful inversion of Graphical Models for Effective
# Amortized Inference: https://proceedings.neurips.cc/paper/2018/file/894b77f805bd94d292574c38c5d628d5-Paper.pdf 
"""
Different functionalities related to a binary-tree-shaped linear 
Gaussian network.
"""

import torch
import numpy as np
import graph_tool.all as gt
import graph.invert as invert

from torch.distributions import Normal
from math import sqrt
from functools import reduce

from graph.belief_network import BeliefNetwork
from graph.forward_graph import ForwardGraph


class GaussianBinaryTree(BeliefNetwork):

    def __init__(self, depth, coeffs, root=(0.0, 1.0), variances=None):
        super(GaussianBinaryTree, self).__init__() 
        self.depth = depth
        self.coeffs = torch.FloatTensor(coeffs)
        self.root = root

        self.forward_graph = ForwardGraph().initialize(*GaussianBinaryTree.construct_tree(depth))
        self.inverse_graph = invert.properly(self.forward_graph)

        self.n = len(self.forward_graph.get_vertices())
        self.num_latent = (self.n+1)//2 - 1

        self.topological_order = list(filter(lambda x: x < self.num_latent, 
            list(gt.topological_sort(self.inverse_graph))))

        if variances is None:
           self.variances = np.ones((self.n-1)) 

    
    def get_num_latent(self):
        return self.num_latent

    
    def get_num_obs(self):
        return self.n - self.num_latent

    
    def get_num_vertices(self):
        return self.n

    
    def topological_order(self):
        return gt.topological_sort(self.forward_graph)


    def inv_topological_order(self):
        return gt.topological_sort(self.inverse_graph)

    
    def sample(self, batch_size=250, train=True):
        # Create root node
        root_dist = Normal(
            loc=torch.tensor([self.root[0]]), 
            scale=torch.tensor([sqrt(self.root[1])]))
        root = root_dist.sample(sample_shape=(batch_size,))

        # Create nodes below the root
        nodes = [root]
        for i in range(1, self.n):
            means = torch.add(
                torch.mul(nodes[self.parent(i)], self.coeffs[i-1,1]),
                self.coeffs[i-1,0])
            nodes.append(Normal(
                loc=means, 
                scale=torch.mul(torch.ones_like(means), sqrt(self.variances[i-1]))
                ).sample())

        samples = torch.cat(nodes, dim=1)
        return samples


    def log_likelihood(self, x, z):
        # Compute p(x|z)
        device = x.device
        sample = torch.cat((z, x), dim=1)
        coeffs = self.coeffs.to(device)
        log_prob = 0.0

        for i in range(self.num_latent, self.n):
            j = self.parent(i)
            means = torch.add(
                torch.mul(sample[:,j:j+1], coeffs[i-1,1]),
                coeffs[i-1,0])
            node_dist = Normal(
                loc=means, 
                scale=torch.mul(
                    torch.ones_like(means),
                    sqrt(self.variances[i-1])))
            log_prob += node_dist.log_prob(sample[:,i:i+1])

        return log_prob


    def log_prior(self, z):
        # Compute p(z)
        device = z.device

        # Create root node
        root_dist = Normal(
            loc=torch.tensor([self.root[0]]).to(device), 
            scale=torch.tensor([sqrt(self.root[1])]).to(device))
        log_prob = root_dist.log_prob(z[:,0:1])

        # Create nodes below the root
        coeffs = self.coeffs.to(device)
        for i in range(1,self.num_latent):
            j = self.parent(i)
            means = torch.add(
                torch.mul(z[:,j:j+1], coeffs[i-1,1]),
                coeffs[i-1,0])
            node_dist = Normal(
                loc=means, 
                scale=torch.mul(
                    torch.ones_like(means),  
                    sqrt(self.variances[i-1])))
            log_prob += node_dist.log_prob(z[:,i:i+1])

        return log_prob

    
    def log_joint(self, x, z):
        log_lik = self.log_likelihood(x, z)
        log_prior = self.log_prior(z)
        return log_lik + log_prior


    def posterior_parameters(self):
        """Compute the parameters (beta, sigma2) of the posterior conditional 
        distribution of each latent variable and the mean and covariance 
        matrix (mu, sigma) of the global joint distribution.

        Returns
        -------
        beta0 : list of numpy.ndarray
        beta : list of numpy.ndarray
            The coefficients used in the calculation of the mean parameter of each
            conditional in the inverse graph
        sigma2 : list of numpy.ndarray
            The variance parameter of each conditional in the inverse graph
        mu : numpy.ndarray
            The mean of the joint distribution
        sigma : numpy.ndarray
            The covariance matrix of the joint distribution of all variables
        """

        # ---------------------------------------------------------------
        # Calculate parameters of the multi-variate normal joint 
        # distribution p(x1,..,Xn):
        # mu_i = coeff_i0 + coeff_i * mu_pa(i),    for i = 1,..,n
        # sigma_ii = sigma_i^2 + coeff_i^2 * sigma_pa(i)^2
        # sigma_ij = 0                          if j not parent(i)
        #          = coeff_j * sigma_pa(i)^2    else
        # ---------------------------------------------------------------
        mu = np.zeros((self.n))
        sigma = np.zeros((self.n,self.n))
        mu[0] = self.root[0]
        sigma[0,0] = self.root[1]
        for i in range(1,self.n):
            j = self.parent(i)
            mu[i] = self.coeffs[i-1,0] + self.coeffs[i-1,1] * mu[j]
            sigma[i,i] = self.variances[i-1] + (self.coeffs[i-1,1]**2 * sigma[j,j])
            
        for k in range(self.n):
            for i in range(k+1,self.n):    
                sigma[k,i] = self.coeffs[i-1,1] * sigma[self.parent(i),k]
                sigma[i,k] = sigma[k,i]

        # ---------------------------------------------------------------
        # Calculate posterior conditional distributions given the joint
        # distribution p(x) = norm(mu, sigma):
        # p(y|x) = norm(beta0 + beta.T*x, sigma2)
        # where
        # beta0 = mu_y - sigma_yx * inv(sigma_xx) * mu_x
        # beta = inv(sigma_xx) * sigma_yx
        # sigma2 = sigma_yy - sigma_yx * inv(sigma_xx) * sigma_yx
        # ---------------------------------------------------------------
        beta0 = []
        beta = []
        sigma2 = []

        for i in range(self.num_latent):
            Y = i
            X = sorted(self.inverse_graph.get_in_neighbours(i))
            id_y = np.array([Y])
            id_x = np.array(X)

            sigma_xx = sigma[id_x[:,None], id_x]
            sigma_xy = sigma[id_x[:,None], id_y]
            sigma_yx = sigma[id_y[:,None], id_x]
            sigma_yy = sigma[Y,Y]
            mu_x = mu[X]
            mu_y = mu[Y]
            sigma_xx_inv = np.linalg.inv(sigma_xx)
            
            beta0.append(np.float64(mu_y - sigma_yx.dot(sigma_xx_inv).dot(mu_x)))
            beta.append(torch.DoubleTensor(sigma_xx_inv.dot(sigma_yx.T)))
            sigma2.append(np.float64(sigma_yy - sigma_yx.dot(sigma_xx_inv).dot(sigma_xy)))
        
        return beta0, beta, sigma2, mu, sigma


    def sample_posterior(self, x):
        """Generate samples of the latent variables.

        Parameters
        ----------
        x : torch.Tensor
            batch_size samples of observed variables from the forward model

        Returns
        -------
        z : torch.tensor
            batch_size sample of latent variables generated from the
            inverse graph given the observed variables x
        """
        nz = self.num_latent
        device = x.device
        beta0, beta, sigma2, _, _ = self.posterior_parameters()

        z = [None for i in range(nz)]

        for i in self.topological_order:
            parents = sorted(self.inverse_graph.get_in_neighbours(i).tolist())

            # sample latent variable i
            parents_sample = torch.cat([
                z[i] if i < nz else 
                x[:,(i-nz):(i-nz)+1]
                for i in parents], dim=1).to(device)
            sample_means = torch.add(
                torch.matmul(parents_sample, beta[i].to(device)),
                beta0[i])
            sample_dists = Normal(loc=sample_means, scale=sqrt(sigma2[i]))
            z[i] = sample_dists.sample()
        
        z = torch.cat(z, dim=1)
        return z


    def log_posterior(self, samples):
        """Calculate the log-posterior q(z|x) of the given samples.

        Parameters
        ----------
        samples : torch.Tensor
            batch_size samples 

        Returns
        -------
        log_likelihood : float
        """
        device = samples.device
        beta0, beta, sigma2, _, _ = self.posterior_parameters()

        log_probs = []

        for i in self.topological_order:
            parents = sorted(self.inverse_graph.get_in_neighbours(i).tolist())

            # Calculate log-likelihood log(p(z|pa(z)))
            parents_samples =  torch.cat([samples[:,i:i+1] for i in parents], dim=1)
            means = torch.add(
                torch.matmul(parents_samples, beta[i].to(device)),
                beta0[i])
            dists = Normal(loc=means, scale=sqrt(sigma2[i]))
            log_probs.append(torch.mean(
                dists.log_prob(samples[:,i:i+1])
            ))
        
        # log(p(z|x)) = sum{ log(p(z_i|pa(z_i))) }
        log_likelihood = reduce((lambda x, y: x + y), log_probs)
        return log_likelihood

    
    def parent(self, id):
        """Returns the id of the parent of the node"""
        return (id-1)//2


    @staticmethod
    def construct_tree(depth):
        """Construct binary tree of specified depth
    
        Parameters
        ----------
        depth : int, optional

        Returns
        -------
        vertices : list of str
        edges : list of str
        observes : list of str
        """

        vertices = []
        for d in range(depth):
            for i in range(2**d):
                vertices.append('x_{}_{}'.format(d, i))
        
        edges = []
        for d in range(depth-1):
            for i in range(2**(d+1)):
                edges.append(('x_{}_{}'.format(d, i//2), 'x_{}_{}'.format(d+1, i)))

        observed = set()
        for i in range(2**(depth-1)):
            observed.add('x_{}_{}'.format(depth-1, i))
        
        return vertices, edges, observed


    @staticmethod
    def rand_coeffs(_rnd, depth):
        n = 2**depth - 1
        return _rnd.uniform(low=0.5, high=2.0, size=(n-1, 2))