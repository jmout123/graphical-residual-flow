# From: Faithful Inversion of Generative Models for Effective Amortized 
# Inference: https://proceedings.neurips.cc/paper/2018/file/894b77f805bd94d292574c38c5d628d5-Paper.pdf 
#
# MIT License
#
# Copyright (c) 2017, Probabilistic Programming Group at University of Oxford
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import sys
import graph_tool.all as gt


def properly(g):
	"""Compute minimally faithful inverse of graph g

	Parameters
	----------
	g : graph.ForwardGraph
		The graphical model to invert
		
	Returns
	-------
	inverse_g : graph_tool.all.Graph
		The minimally faithful inverse of g
	"""

	# Array to keep track of vertices (ones that have been eliminated in simulation of VE)
	marked = np.zeros(g.num_vertices())

	# Start off with just the vertices of g
	inverse_g = copy_vertices(g)

	# Moralize G, and simulate variable elimination according to the topological ordering, building the implicit (?) graph
	moral_g = moralize(g)
	names = g.vertex_properties['name']

	# Frontier of nodes that can be chosen next in the topological ordering
	count_parents = g.get_in_degrees(g.get_vertices())
	frontier = set(np.where(count_parents == 0)[0])

	# Remove evidence nodes from frontier
	remove_list = []
	for v_idx in frontier:
		if v_idx in g.observed:
			remove_list.append(v_idx)

	for v_idx in remove_list:
		frontier.remove(v_idx)

	# Loop while there are elements to add to the topological ordering
	while frontier:
		# Find the vertex according to the greedy min-fill edge heuristic
		min_cost = sys.maxsize
		min_idx = {}
		for v_idx in frontier:
			ns_idx = unmarked_neighbours(moral_g, v_idx, marked)
			cost_v = count_fill_edges(moral_g, ns_idx, marked)

			# Take the lowest cost vertex that also has the least order lexicographically
			if cost_v < min_cost:
				min_cost = cost_v
				min_idx = {v_idx}
			elif cost_v == min_cost:
				min_idx.add(v_idx)

		min_names = sorted([g.vertex_properties['name'][g.vertex(v)] for v in min_idx])
		min_idx = int(g.vertex_name_map[min_names[0]])

		# Remove from S
		v_idx = min_idx
		frontier.remove(v_idx)

		# Add edges between non-marked neighbours, the covering edges
		ns_idx = unmarked_neighbours(moral_g, v_idx, marked)

		fill_edges(moral_g, ns_idx, marked)

		# Mark v
		marked[v_idx] = 1

		# Make unmarked neighbours of v in moral_g the parents of v in inverse_g
		# For each unmarked child u of v in G, add it to frontier if all parents marked
		for u_idx in ns_idx:
			inverse_g.add_edge(u_idx, v_idx)
			if not u_idx in g.observed:
				if all_parents_marked(g, u_idx, marked, g.observed):
					frontier.add(u_idx)

	return inverse_g


def properly_fixed_order(g, order):
	"""Compute faithful inverse of graph g given a variable elimination order

	Parameters
	----------
	g : graph.ForwardGraph
		The graphical model to invert
	order : list
		A list of variable node indices
		
	Returns
	-------
	inverse_g : graph_tool.all.Graph
		The minimally faithful inverse of g
	"""

	# Array to keep track of vertices (ones that have been eliminated in simulation of VE)
	marked = np.zeros(g.num_vertices())

	# Start off with just the vertices of g
	inverse_g = copy_vertices(g)

	# Moralize G, and simulate variable elimination according to the topological ordering, building the implicit (?) graph
	moral_g = moralize(g)
	names = g.vertex_properties['name']

	# Loop while there are elements to add to the topological ordering
	for idx in order:
		v_idx = idx

		# Add edges between non-marked neighbours, the covering edges
		ns_idx = unmarked_neighbours(moral_g, v_idx, marked)

		fill_edges(moral_g, ns_idx, marked)

		# Mark v
		marked[v_idx] = 1

		# Make unmarked neighbours of v in moral_g the parents of v in inverse_g
		# For each unmarked child u of v in G, add it to frontier if all parents marked
		for u_idx in ns_idx:
			inverse_g.add_edge(u_idx, v_idx)
			
	return inverse_g


def copy_vertices(g):
	"""Create a graph that has all the vertices of g but none of the edges

	Parameters
	----------
	g : graph_tool.all.Graph
		Graph whose vertices to extract

	Returns
	-------
	new_g : graph_tool.all.Graph
		Graph like g with no edges
	"""

	new_g = gt.Graph(g)
	new_g.clear_edges()
	return new_g


def moralize(g):
	"""Moralize a directed graph by connecting all the parents, and converting to an undirected graph

	Parameters
	----------
	g : graph_tool.all.Graph
		Directed graph

	Returns
	-------
	moral_g : graph_tool.all.Graph
		Moralized undirected graph
	"""

	# Make copy of g
	moral_g = gt.Graph(g)
	moral_g.set_directed(False)
	names = g.vertex_properties['name']

	# TODO: Connect all parents of g

	# Iterate over all vertices
	for v_idx in g.get_vertices():
		v = g.vertex(v_idx)

		parents = g.get_in_neighbours(v_idx)

		# Enumerate over all potential edges between parents of vertex
		for idx, w_idx in enumerate(parents):
			out_edges = g.get_out_edges(w_idx)

			for u_idx in parents[(idx+1):]:
				in_edges = g.get_out_edges(u_idx)

				# Test if u-w or w-u exists and if not, create u-w in moralized graph
				if not (u_idx in out_edges[:, 1] or w_idx in in_edges[:, 1]):
					# Check that haven't added already to moral_g
					out_edges_moral = moral_g.get_out_edges(w_idx)
					in_edges_moral = moral_g.get_out_edges(u_idx)

					if not (u_idx in out_edges_moral[:, 1] or w_idx in in_edges_moral[:, 1]):
						moral_g.add_edge(u_idx, w_idx)
	return moral_g


def unmarked_neighbours(g, v_idx, marked):
	"""Build list of non-eliminated neighbours of v

	Parameters
	 ----------
	g : graph_tool.all.Graph
		 The graph
	v_idx : int
		The vertex of interest
	marked : list of bool
		Indication of which vertices are marked

	Returns
	-------
	scope : list of int
		List of unmarked neighbours
	"""

	neighbours = g.get_out_neighbours(v_idx)

	scope = []
	for u_idx in neighbours:
		if not marked[u_idx]:
			scope.append(u_idx)
	return scope


def all_parents_marked(g, v_idx, marked, observed):
	"""Determine whether all of v's latent parents are marked."""
	parents = g.get_in_neighbours(v_idx)
	for u_idx in parents:
		# Return false if a parent is both unmarked and a latent variable
		if (not marked[u_idx]) and (u_idx not in observed):
			return False
	return True


def count_fill_edges(g, neighbours_idx, marked):
	count = 0
	for idx, u_idx in enumerate(neighbours_idx):
		u_neighbours = g.get_out_neighbours(u_idx)
		for w_idx in neighbours_idx[(idx+1):]:
			if not w_idx in u_neighbours:
				count += 1
	return count


def fill_edges(g, neighbours_idx, marked):
	for idx, u_idx in enumerate(neighbours_idx):
		u_neighbours = g.get_out_neighbours(u_idx)
		for w_idx in neighbours_idx[(idx+1):]:
			if not w_idx in u_neighbours:
				g.add_edge(u_idx, w_idx)


def markov_blanket(g, v_idx):
	"""Compute markov blanket of given vertex

	Parameters
	----------
	g : graph_tool.all.Graph
		The graphical model
	v_idx : int
		Index of the vertex
		
	Returns
	-------
	parents_of_children : np.ndarray
		Array of indices of markov blanket
	"""

	parents = g.get_in_neighbours(v_idx)
	children = g.get_out_neighbours(v_idx)

	parents_of_children = set()
	for u_idx in children:
		parents_of_children.update(g.get_in_neighbours(u_idx))
	parents_of_children.discard(v_idx)

	parents_of_children.update(parents)

	parents_of_children.update(children)

	return np.asarray(list(parents_of_children))