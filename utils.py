# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 21:36:51 2022

@author: Costelinha
"""
import numpy as np

def Average_degree(G):
    degrees = [G.degree(n) for n in G.nodes()]
    mean=np.mean(degrees)
    return mean