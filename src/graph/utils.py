# From: Faithful Inversion of Generative Models for Effective Amortized 
# Inference with additions: https://proceedings.neurips.cc/paper/2018/file/894b77f805bd94d292574c38c5d628d5-Paper.pdf 

import numpy as np
import graph_tool.all as gt

from warnings import warn
from math import ceil


def draw(g, name="", pos=None, output_size=(600, 600)):
	gt.graph_draw(
		g,
		pos = pos,
		vertex_text=g.vertex_properties['name'],
		output_size=output_size,
		fit_view=True,
		output=name+'.png',
		bg_color=[1., 1., 1., 1.],
		vertex_color=[0., 0., 0., 1.],
		vertex_fill_color=[1., 1., 1., 1.],
		vertex_text_color=[0., 0., 0., 1.],
		vertex_pen_width=0.8,
		edge_marker_size=10
		)


def adjacency_matrix(g):
	"""Construct adjacency matrix for graph g

	Parameters
	----------
	g : graph_tool.all.Graph

	Returns
	-------
	A : numpy.ndarray
	"""
	n = g.num_vertices()
	A = np.zeros((n, n), dtype=np.float64)
	for from_idx in g.get_vertices():
		A[from_idx, from_idx] = 1
		for to_idx in g.get_out_neighbours(from_idx):
			A[to_idx, from_idx] = 1
	
	return A


def graph_masks(g, input_vars, output_vars, hidden_dims, 
				self_dependent=False):
	"""Construct masks to control information flow between layers
	of neural network.
	
	Based on MADE approach used for density estimation, where random
	variables are assumed to be fully connected.

	Construct len(hidden_dims) + 1 masks, such that each 
	variable in output_vars is connected through the hidden 
	layers to only those variables in the inputs_vars on which
	it is conditionally dependent given the graph g.
	"""
	# NOTE: input_vars, output_vars need to be in the order they
	# will be used in the neural network

	masks = []

	# Create subsets
	# Input
	input_subsets = []
	for x in input_vars:
		input_subsets.append({x})
	
	# Hidden
	hidden_subsets = []
	parents = set() 
	for x in set(output_vars):
		p = g.get_in_neighbours(x)
		parents |= {*p}
		if self_dependent:
			parents |= {x}
		if len(p) >= 1:
			if self_dependent:
				hidden_subsets.append({*p} | {x})
			else:
				hidden_subsets.append({*p})
	for x in parents:
		hidden_subsets.append({x})

	# Output
	output_subsets = []
	for x in output_vars:
		parents = g.get_in_neighbours(x)
		if self_dependent:
			output_subsets.append({*parents} | {x})
		else:
			output_subsets.append({*parents})

	for h in hidden_dims:
		if h < len(hidden_subsets):
			warn("Size of hidden layer ({}) is less than number of subsets that need to be assigned for mask creation ({}).".format(h, len(hidden_subsets)))

	# Assign hidden subset indices to units of each hidden layer
	hidden_subset_idxs = [
		([*np.arange(0,len(hidden_subsets))]*(ceil(hidden_dims[i]/len(hidden_subsets))))[:hidden_dims[i]]
		for i in range(len(hidden_dims))
	]

	# Create mask between input layer and first hidden layer
	hidden_dim = hidden_dims[0]
	in_dim = len(input_vars)
	hidden_subset_idx = hidden_subset_idxs[0]
	mask = np.zeros((hidden_dim, in_dim), dtype=np.float64)
	
	for h in range(hidden_dim):
		h_set = hidden_subsets[hidden_subset_idx[h]]
		for i in range(in_dim):
			if h_set.issuperset(input_subsets[i]):
				mask[h, i] = 1
	masks.append(mask)

	# Create mask between hidden layers
	for l in range(len(hidden_dims)-1):
		hidden_dim1 = hidden_dims[l]
		hidden_dim2 = hidden_dims[l+1]
		h_subset_idx1 = hidden_subset_idxs[l]
		h_subset_idx2 = hidden_subset_idxs[l+1]
		mask = np.zeros((hidden_dim2, hidden_dim1), dtype=np.float64)

		for h2 in range(hidden_dim2):
			h2_set = hidden_subsets[h_subset_idx2[h2]]
			for h1 in range(hidden_dim1):
				h1_set = hidden_subsets[h_subset_idx1[h1]]
				if h2_set.issuperset(h1_set):
					mask[h2, h1] = 1
		masks.append(mask)

	# Create mask between last hidden layer and output layer
	hidden_dim = hidden_dims[-1]
	out_dim = len(output_vars)
	hidden_subset_idx = hidden_subset_idxs[-1]
	mask = np.zeros((out_dim, hidden_dim), dtype=np.float64)
	
	for h in range(hidden_dim):
		h_set = hidden_subsets[hidden_subset_idx[h]]
		for i in range(out_dim):
			if output_subsets[i].issuperset(h_set):
				mask[i, h] = 1
	masks.append(mask)

	return masks


def autoregressive_masks(in_dim, out_dim, num_out_chunks, hidden_dims):
	masks = []

	# Create subsets
	input_indices = np.arange(in_dim).tolist()
	hidden_indices = np.arange(1, in_dim).tolist()
	output_indices = (np.arange(out_dim).tolist())*num_out_chunks

	# Randomly assign index to each unit of each 
	# hidden layer
	hidden_idx_assignments = [
		np.random.choice(hidden_indices, size = hidden_dims[i]) for i in range(len(hidden_dims))
	]

	# Create mask between input layer and first hidden layer
	hidden_dim = hidden_dims[0]
	hidden_idx_assignment = hidden_idx_assignments[0]
	mask = np.zeros((hidden_dim, in_dim), dtype=np.uint8)
	
	for h in range(hidden_dim):
		h_idx = hidden_idx_assignment[h]
		for i in input_indices:
			if i >= h_idx:
				mask[h, i] = 1
	masks.append(mask)

	# Create mask between hidden layers
	for l in range(len(hidden_dims)-1):
		hidden_dim1 = hidden_dims[l]
		hidden_dim2 = hidden_dims[l+1]
		h_idx_assignment1 = hidden_idx_assignments[l]
		h_idx_assignment2 = hidden_idx_assignments[l+1]
		mask = np.zeros((hidden_dim2, hidden_dim1), dtype=np.uint8)

		for h2 in range(hidden_dim2):
			h2_idx = h_idx_assignment2[h2]
			for h1 in range(hidden_dim1):
				h1_idx = h_idx_assignment1[h1]
				if h1_idx >= h2_idx:
					mask[h2, h1] = 1
		masks.append(mask)

	# Create mask between last hidden layer and output layer
	hidden_dim = hidden_dims[-1]
	hidden_idx_assignment = hidden_idx_assignments[-1]
	mask = np.zeros((out_dim*num_out_chunks, hidden_dim), dtype=np.uint8)
	
	for h in range(hidden_dim):
		h_idx = hidden_idx_assignment[h]
		for i in output_indices:
			if h_idx > i:
				mask[i, h] = 1
	masks.append(mask)

	return masks