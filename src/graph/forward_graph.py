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


import graph_tool.all as gt
import graph.utils


class ForwardGraph(gt.Graph):
	"""Take graph information and construct a graph_tool graph."""

	def __init__(self):
		super(ForwardGraph, self).__init__()


	def initialize(self, vertices, edges, observes):
		# Create a property for the vertex names
		self.vertex_properties['name'] = self.new_vertex_property('string')

		# Construct vertices
		self.add_vertex(len(vertices))
		self.vertex_name_map = {}

		for idx, v in enumerate(self.vertices()):
			self.vertex_properties['name'][v] = vertices[idx]
			self.vertex_name_map[vertices[idx]] = v

		# Construct edges
		for s, d in edges:
			self.add_edge(self.vertex_name_map[s], self.vertex_name_map[d])

		# Convert observes to a vertex index
		self.observe_names = observes
		self.observed = set([self.vertex_index[self.vertex_name_map[n]] for n in observes])

		return self


	def print_factors(self):
		edges = self.get_edges()
		factor_scopes = {}
		for idx in range(edges.shape[0]):
			from_v = self.vertex_properties['name'][self.vertex(edges[idx, 0])]
			to_v = self.vertex_properties['name'][self.vertex(edges[idx, 1])]

			if to_v not in factor_scopes:
				factor_scopes[to_v] = {from_v}
			else:
				factor_scopes[to_v].add(from_v)

		for k in sorted(factor_scopes.keys()):
			print('{} | {}'.format(k, sorted(factor_scopes[k])))


	def print_edges(self):
		edges = self.get_edges()
		for idx in range(edges.shape[0]):
			from_v = self.vertex_properties['name'][self.vertex(edges[idx, 0])]
			to_v = self.vertex_properties['name'][self.vertex(edges[idx, 1])]
			print('{} -> {}'.format(from_v, to_v))