import torch
import numpy as np
import graph_tool.all as gt
import graph.invert as invert

from torch.distributions import Normal, Laplace

from graph.forward_graph import ForwardGraph
from graph.belief_network import BeliefNetwork


class ArithmeticCircuit(BeliefNetwork):

    def __init__(self):
        super(ArithmeticCircuit, self).__init__()        
        # Node distributions
        self.z0 = Laplace(5,1)
        self.z1 = Laplace(-2,1)
        self.z2 = self.z2tmp
        self.z3 = self.z3tmp
        self.z4 = Normal(7,2)
        self.z5 = self.z5tmp

        self.x0 = self.x0tmp
        self.x1 = self.x1tmp

        # graph
        self.forward_graph = ForwardGraph().initialize(*ArithmeticCircuit._construct_graph())
        self.inverse_graph = invert.properly(self.forward_graph)

        self.num_latent = 6
        self.num_obs = 2


    def z2tmp(self, z0, z1):
        return Normal(torch.tanh(z0 + z1 - 2.8), 0.1)

    def z3tmp(self, z0, z1): 
        return Normal(z0*z1, 0.1)

    def z5tmp(self, z3, z4): 
        return Normal(torch.tanh(z3 + z4), 0.1)

    def x0tmp(self, z3): 
        return Normal(z3, 0.1)

    def x1tmp(self, z5): 
        return Normal(z5, 0.1)


    def sample(self, batch_size, train=True):
        z0 = self.z0.sample((batch_size,1))
        z1 = self.z1.sample((batch_size,1))
        z2 = self.z2(z0, z1).sample()
        z3 = self.z3(z0, z1).sample()
        z4 = self.z4.sample((batch_size,1))
        z5 = self.z5(z3, z4).sample()

        x0 = self.x0(z3).sample()
        x1 = self.x1(z5).sample()

        sample = torch.cat([z0,z1,z2,z3,z4,z5,x0,x1], dim=1)
        return sample


    def log_likelihood(self, x, z):
        # p(x|z) = p(x0|z3)p(x1|z5)
        assert x.shape[1] == 2
        log_p_x0 = self.x0(z[:,3]).log_prob(x[:,0])
        log_p_x1 = self.x1(z[:,5]).log_prob(x[:,1])
        return log_p_x0 + log_p_x1


    def log_prior(self, z):
        # p(z) = p(z0) x p(z1) x p(z2|z0,z1) x p(z3|z0,z1) x
        #            p(z4) x p(z5|z3,z4)
        log_p_z = self.z0.log_prob(z[:,0]) 
        log_p_z += self.z1.log_prob(z[:,1])
        log_p_z += self.z2(z[:,0], z[:,1]).log_prob(z[:,2]) 
        log_p_z += self.z3(z[:,0], z[:,1]).log_prob(z[:,3]) 
        log_p_z += self.z4.log_prob(z[:,4]) 
        log_p_z += self.z5(z[:,3], z[:,4]).log_prob(z[:,5])

        return log_p_z 


    def log_joint(self, x, z):
        # p(x,z) = p(x|z)p(z)
        log_lik = self.log_likelihood(x, z)
        log_prior = self.log_prior(z)
        return log_lik + log_prior

    
    def sample_base_prior(self, batch_size):
        z0 = Normal(0,1).sample((batch_size,1))
        z1 = Normal(0,1).sample((batch_size,1))
        z2 = Normal(0,1).sample((batch_size,1))
        z3 = Normal(0,1).sample((batch_size,1))
        z4 = Normal(0,1).sample((batch_size,1))
        z5 = Normal(0,1).sample((batch_size,1))

        return torch.cat([z0,z1,z2,z3,z4,z5], dim=1)

    def log_base_prior(self, z):
        # p(z) = p(z0) x p(z1) x p(z2|z0,z1) x p(z3|z0,z1) x
        #            p(z4) x p(z5|z3,z4)
        log_p_z = Normal(0,1).log_prob(z[:,0])
        log_p_z += Normal(0,1).log_prob(z[:,1])
        log_p_z += Normal(0,1).log_prob(z[:,2])
        log_p_z += Normal(0,1).log_prob(z[:,3])
        log_p_z += Normal(0,1).log_prob(z[:,4])
        log_p_z += Normal(0,1).log_prob(z[:,5])

        return log_p_z 


    def get_num_latent(self):
        return 6


    def get_num_obs(self):
        return 2


    def get_num_vertices(self):
        return 8


    def topological_order(self):
        return gt.topological_sort(self.forward_graph)


    def inv_topological_order(self):
        return gt.topological_sort(self.inverse_graph) 


    @staticmethod
    def _construct_graph():
        vertices = ['z0', 'z1', 'z2', 'z3', 'z4', 'z5', 
                    'x0', 'x1']
        edges = [('z0','z2'), ('z0','z3'), ('z1','z2'),
                 ('z1','z3'), ('z3','z5'), ('z3','x0'), 
                 ('z4','z5'), ('z5','x1')]  
        observed = {'x0', 'x1'}
        return vertices, edges, observed