#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import math
import time
import torch
import numpy as np 
import pandas as pd
import networkx as nx
from scipy import spatial
import matplotlib.pyplot as plt

class GrowingNeuralGas():
    def __init__(self, input_data):
        self.network = None
        self.data = input_data
        self.units_created = 0
        plt.style.use('ggplot')
        self.l = 100
        self.passes=0
        self.accumulated_local_error = []
        self.global_error = []
        self.network_order = []
        self.network_size = []
        self.total_units = []
        self.plot_evolution = True
        self.sequence = 0  
        self.shapemat = 0
        self.steps = 0
        self.nodedel = 0
        self.global_error = 0
        self.larr = []
        self.counter = 0
        self.cn = 0
 
    def plot_network(self, file_path):
        plt.clf()
        plt.scatter(self.data[:, 0], self.data[:, 1])
        node_pos = {}
        for u in self.network.nodes():
            vector = self.network.nodes[u]['vector']
            node_pos[u] = (vector[0], vector[1])
        nx.draw(self.network, pos=node_pos)
        plt.draw()
        plt.savefig(file_path)


    def find_nearest_units(self, observation):
        distance = []
#         print("len", len(list(self.network.nodes(data=True))))
        for u, attributes in list(self.network.nodes(data=True)):
            vector = attributes['vector']
            dist = spatial.distance.euclidean(vector, observation)
            distance.append((u, dist))
        distance.sort(key=lambda x: x[1])
        ranking = [u for u, dist in distance]
        #print(ranking)
        #print("dist",dist)
        return ranking


    def GNG_adapt(self, d, age_max, e_b, e_n, observation):
        nearest_units = self.find_nearest_units(observation)
        s_1 = nearest_units[0]
        s_2 = nearest_units[1]
#         print(s_1)
#         print(s_2)

        # 4. add the squared distance between the observation and the nearest unit in input space
        self.network.nodes[s_1]['error'] += spatial.distance.euclidean(observation, self.network.nodes[s_1]['vector'])**2


        # 5 .move s_1 and its direct topological neighbors towards the observation by the fractions
        #    e_b and e_n, respectively, of the total distance
        update_w_s_1 = e_b * (np.subtract(observation, self.network.nodes[s_1]['vector']))
        self.network.nodes[s_1]['vector'] = np.add(self.network.nodes[s_1]['vector'], update_w_s_1)
        update_w_s_n = e_n * (np.subtract(observation, self.network.nodes[s_1]['vector']))
        for neighbor in list(self.network.neighbors(s_1)):
            self.network.nodes[neighbor]['vector'] = np.add(self.network.nodes[neighbor]['vector'], update_w_s_n)


        # 6. if s_1 and s_2 are connected by an edge, set the age of this edge to zero
        #    if such an edge doesn't exist, create it

        self.network.add_edge(s_1, s_2, age=0)

        for u, v, attributes in list(self.network.edges(data=True, nbunch=[s_1])):
            self.network.add_edge(u, v, age=attributes['age']+1)


        # 7. remove edges with an age larger than age_max
        #    if this results in units having no emanating edges, remove them as well

        self.prune_connections(age_max)
        error = 0
        for u in list(self.network.nodes()):
            error += self.network.nodes[u]['error']
        self.accumulated_local_error.append(error)
        self.network_order.append(self.network.order())
        self.network_size.append(self.network.size())
        self.total_units.append(self.units_created)
        #       print(list(self.network.nodes(data=True)))

        for u in list(self.network.nodes()):
            self.network.nodes[u]['error'] *= d
            if self.network.degree(nbunch=[u]) == 0:
                print(u)
#         self.global_error.append(self.compute_global_error())
        #print("Total steps are", steps)

        return list(self.network.nodes(data=True))


    def GNG_newnode(self, plot_evolution,a):
        # 8.a determine the unit q with the maximum accumulated error
        q = 0

        error_max = 0
#         print(list(self.network.nodes(data = True)))
        for u in list(self.network.nodes()):
            if self.network.nodes[u]['error'] > error_max:
                error_max = self.network.nodes[u]['error']
                q = u
                #print("emax",error_max)
        # 8.b insert a new unit r halfway between q and its neighbor f with the largest error variable
        #print("q is",q)
        f = -1
        largest_error = -1
        #print(list(self.network.nodes(data=True)))
        for u in list(self.network.neighbors(q)):
            if self.network.nodes[u]['error'] > largest_error:
                largest_error = self.network.nodes[u]['error']
                f = u
                #print("lerr",largest_error)
        w_r = 0.5 * (np.add(self.network.nodes[q]['vector'], self.network.nodes[f]['vector']))
        #print("wr",w_r)
        #print(list(self.network.nodes(data=True)))
        r = self.units_created
        #print("r",r)
        self.units_created += 1
        # 8.c insert edges connecting the new unit r with q and f
        #     remove the original edge between q and f
        self.network.add_node(r, vector=w_r, error=0)
        self.network.add_edge(r, q, age=0)
        self.network.add_edge(r, f, age=0)
        self.network.remove_edge(q, f)
        # 8.d decrease the error variables of q and f by multiplying them with a
        #     initialize the error variable of r with the new value of the error variable of q
        self.network.nodes[q]['error'] *= a
        self.network.nodes[f]['error'] *= a
        self.network.nodes[r]['error'] = self.network.nodes[q]['error']
        # if self.plot_evolution:
        #   self.plot_network('/content/' + str(self.sequence) + '.png')
        #   self.sequence += 1
        #print(list(self.network.nodes(data=True)))
        return list(self.network.nodes(data=True))



    def GNG(self, e_b, e_n, age_max, a, d, ncol, nrow, l, num_nodes, plot_evolution=True):
        start = time.time()
        num_nodes = num_nodes
#         print("Total nodes should be", num_nodes)
#         print("Total Data rows, lambda", nrow, l)
          
        # 0. start with two units a and b at random position w_a and w_b
        w_a = [np.random.uniform(-2, 2) for _ in range(ncol)]
        w_b = [np.random.uniform(-2, 2) for _ in range(ncol)]
        
        self.network = nx.Graph()
        self.network.add_node(self.units_created, vector=w_a, error=0)
        self.units_created += 1
        self.network.add_node(self.units_created, vector=w_b, error=0)
        self.units_created += 1
#         print("initially", self.network.number_of_nodes())

        # 1. iterate through the data
        while self.network.number_of_nodes() <= num_nodes:
#             print("Number of nodes currently in graph:", self.network.number_of_nodes(), num_nodes)
            np.random.shuffle(self.data)
            observations = self.data[:l]
            for observation in self.data:
                self.GNG_adapt(d, age_max, e_b, e_n, observation)
            self.GNG_newnode(plot_evolution,a)

#         print(self.network.number_of_nodes())
        self.larr= [np.array(list(self.network.nodes(data=True))[i][1]['vector']).ravel() for i in range(len(list(self.network.nodes(data=True))))]
        print("shape is now", np.matrix(self.larr).shape)

        return self.larr
       
     
    def outcsv(self):
        pddata= pd.DataFrame(np.matrix(self.larr),columns =['x'+str(i+1) for i in range(ncol)])
        print("Making GNG output file of shape: ", pddata.shape)
        return pddata



    def prune_connections(self, age_max):
        for u, v, attributes in list(self.network.edges(data=True)):
            if attributes['age'] > age_max:
                self.network.remove_edge(u, v)
        for u in list(self.network.nodes()):
            if self.network.degree(u) == 0:
                self.network.remove_node(u)


    def compute_global_error(self):

        for observation in self.data:
            nearest_units = self.find_nearest_units(observation)
            s_1 = nearest_units[0]
            global_error += spatial.distance.euclidean(observation, self.network.nodes[s_1]['vector'])**2
        return global_error
 


