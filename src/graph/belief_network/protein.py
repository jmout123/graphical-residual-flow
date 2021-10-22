import torch
import numpy as np
import pandas as pd
import os
import graph_tool.all as gt

import graph.invert as invert

from graph.belief_network import BeliefNetwork
from graph.forward_graph import ForwardGraph


class Protein(BeliefNetwork):

    def __init__(self):
        super(Protein, self).__init__()
        # graph
        self.forward_graph = ForwardGraph().initialize(*Protein._construct_graph())
        self.inverse_graph = invert.properly(self.forward_graph)

        self.num_latent = 0
        self.num_obs = 11

        train, test = self.load_data()
        self.train = train.to_numpy()
        np.random.shuffle(self.train)
        self.test = test.to_numpy()
        np.random.shuffle(self.test)
        self.train_idx = 0
        self.test_idx = 0
        self.train_N = train.shape[0]
        self.test_N = test.shape[0]


    def load_data(self):
        dir = './datasets/human_protein/'
        train_path = os.path.join(dir, 'train.csv')
        test_path = os.path.join(dir, 'test.csv')
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        return train, test

    
    def sample(self, batch_size, train=True):
        if train:
            return self._sample_train(batch_size)
        else:
            return self._sample_test(batch_size)


    def _sample_train(self, batch_size):
        sample = self.train.take(
            range(self.train_idx,self.train_idx + batch_size),
            axis=0, mode='wrap')
        self.train_idx = (self.train_idx + batch_size)%self.train_N
        return torch.tensor(sample)


    def _sample_test(self, batch_size):
        sample = self.test.take(
            range(self.test_idx, self.test_idx + batch_size),
            axis=0, mode='wrap')
        self.test_idx = (self.test_idx + batch_size)%self.test_N
        return torch.tensor(sample)


    def log_likelihood(self, x, z):
        raise Exception("Log-likelihood not available for EightPairs BN")


    def log_prior(self, z):
        raise Exception("Log-priors not available for EightPairs BN")


    def log_joint(self, x, z):
        raise Exception("Log-joint not available for EightPairs BN")


    def get_num_latent(self):
        return 0


    def get_num_obs(self):
        return 11


    def get_num_vertices(self):
        return 11


    def topological_order(self):
        return gt.topological_sort(self.forward_graph)


    def inv_topological_order(self):
        return gt.topological_sort(self.inverse_graph) 


    @staticmethod
    def _construct_graph():
        vertices = ['raf', 'mek', 'plcg', 'pip2', 'pip3', 
                    'erk', 'akt', 'pka', 'pkc', 'p38', 'jnk']
        edges = [('raf','mek'),('mek','erk'),('plcg','pip2'),
            ('plcg','pip3'),('plcg','pkc'),('pip2','pkc'), ('pip3','pip2'),('pip3','akt'),('erk','akt'),
            ('pka','raf'),('pka','mek'),('pka','erk'),
            ('pka','akt'),('pka','p38'),('pka','jnk'),
            ('pkc','raf'),('pkc','mek'),('pkc','pka'),
            ('pkc','p38'),('pkc','jnk')]  
        observed = {'raf', 'mek', 'plcg', 'pip2', 'pip3', 
                    'erk', 'akt', 'pka', 'pkc', 'p38', 'jnk'}
        return vertices, edges, observed