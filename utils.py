# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 21:36:51 2022

@author: Costelinha
"""
import numpy as np
import networkx as nx

def Average_degree(G):
    degrees = [G.degree(n) for n in G.nodes()]
    mean=np.mean(degrees)
    return mean

def nodes_connected(G,u, v):
     return u in G.neighbors(v)
 

def average_degree_square(G):
    degrees = [G.degree(n) for n in G.nodes()]
    mean=np.mean(np.square(degrees))
    return mean