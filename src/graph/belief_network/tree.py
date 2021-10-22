import torch
import numpy as np
import sklearn
import graph_tool.all as gt

import graph.invert as invert

from graph.belief_network import BeliefNetwork
from graph.forward_graph import ForwardGraph

class Tree(BeliefNetwork):

    def __init__(self):
        super(Tree, self).__init__()
        # graph
        self.forward_graph = ForwardGraph().initialize(*Tree._construct_graph())
        self.inverse_graph = invert.properly(self.forward_graph)

        self.num_latent = 0
        self.num_obs = 7


    def sample(self, batch_size, train=True):
        rng = np.random.RandomState()

        # Circles
        x0x1 = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        x0x1 *= 3
        x0x1 = x0x1/torch.tensor([1.6944685, 1.6935346])

        # 8 Gaussians
        x2x3 = self.sample_8gaussians(rng, batch_size)
        x2x3 = x2x3/torch.tensor([2.0310535, 2.0305095])

        # x4~ N(max(X0, X1),1)
        x4 = []
        for i in range(batch_size):
            x4.append(np.random.normal(loc=max(x0x1[i,0],x0x1[i,1]),scale=1.0))
        x4 = np.array([x4]).T

        # x5 ~ N(min(X2, X3),1)
        x5 = []
        for i in range(batch_size):
            x5.append(np.random.normal(loc=min(x2x3[i,0],x2x3[i,1]),scale=1.0))
        x5 = np.array([x5]).T

        # x6 ~ 0.5*N(sin(X4+X5),1) + 0.5*N(cos(X4+X5),1).
        x6 = 0.5*np.random.normal(loc=np.sin(x4+x5),scale=np.ones_like(x5)) + \
             0.5*np.random.normal(loc=np.cos(x4+x5),scale=np.ones_like(x5))

        data = np.concatenate([x0x1, x2x3, x4, x5, x6], axis=1)
        data = torch.tensor(data)

        return data


    def sample_8gaussians(self, rng, batch_size):
        scale = 4.
        centers = [(1,0), (-1,0), (0,1), (0,-1), (1./np.sqrt(2), 1./np.sqrt(2)),
            (1./np.sqrt(2), -1./np.sqrt(2)), (-1./np.sqrt(2), 1./np.sqrt(2)), 
            (-1./np.sqrt(2), -1./np.sqrt(2))]
        centers = [(scale*x, scale*y) for x, y in centers]
        data= []
        for i in range(batch_size):
            point = rng.randn(2)*0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            data.append(point)
        data = np.array(data)
        data /= 1.414
        return data


    def log_likelihood(self, x, z):
        raise Exception("Log-likelihood not available for EightPairs BN")


    def log_prior(self, z):
        raise Exception("Log-priors not available for EightPairs BN")


    def log_joint(self, x, z):
        raise Exception("Log-joint not available for EightPairs BN")


    def get_num_latent(self):
        return 0


    def get_num_obs(self):
        return 7


    def get_num_vertices(self):
        return 7


    def topological_order(self):
        return gt.topological_sort(self.forward_graph)


    def inv_topological_order(self):
        return gt.topological_sort(self.inverse_graph) 


    @staticmethod
    def _construct_graph():
        vertices = ['x0','x1', 'x2','x3', 'x4','x5', 'x6']
        edges = [('x1','x0'), ('x2','x1'),('x3','x2'), ('x0','x4'), ('x1','x4'),
                 ('x2','x5'), ('x3','x5'), ('x4','x6'), 
                 ('x5','x6')]  
        observed = {'x0','x1', 'x2','x3', 'x4','x5', 'x6'}
        return vertices, edges, observed