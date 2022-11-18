# -*- coding: utf-8 -*-
"""
@author: Costelinha
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kuramoto import Kuramoto, plot_phase_coherence


class KuramotoPlot:
    
    def __init__(self,G):
        self.G=G
        self.n=G.number_of_nodes()
        

    def plot_order_param_vs_coupling_avg(self, color, nat_freqs, number_of_simulations):
        '''
        Plots the order parameters against the average coupling for a 
        Kuramoto simulation over a graph.
        Inputs:
            self
            color: hex, rgb or string representing the color of the chart
            nat_freqs: TODO distribution of natural frequencies
            number_of_simulations: int representing the total simulations 
                you want to take the average over
        Outputs:
            plot object with the time series of all nodes vs time
        '''
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.6)

        # Initializing graph and arrays
        graph = nx.to_numpy_array(self.G)   
        coupling_vals = np.linspace(0, 1.0, 100)
        mean_of_simulations = np.zeros(len(coupling_vals))
        max_steps = 1000

        for _ in range(number_of_simulations):
            r = np.zeros(len(coupling_vals))
            runs = []
            
            for coupling in coupling_vals:
                model = Kuramoto(coupling=coupling, dt=0.1, T=500, n_nodes=len(graph))
                model.natfreqs =  nat_freqs  # Reset natural frequencies
                act_mat = model.run(adj_mat=graph)
                runs.append(act_mat)
    
            # Check that natural frequencies are correct (we need them for prediction of Kc)
            runs_array = np.array(runs)
    
            for i, coupling in enumerate(coupling_vals):
                r_mean = np.mean( # Mean over last max_steps
                    [
                        model.phase_coherence(vec) for vec in runs_array[i, :, -max_steps:].T
                    ]) 
                r[i] = r_mean
            mean_of_simulations += r / number_of_simulations
    
        # Predicted Kc – analytical result (from paper) - TODO DEPRECATE
        #Kc = np.sqrt(8 / np.pi) * np.std(model.natfreqs) # analytical result (from paper)
        #plt.vlines(Kc, 0, 1, linestyles='--', color='orange', label='analytical prediction')

        # Plotting chart
        np.savetxt(str(self.G),r,fmt='%.2f')
        plt.scatter(coupling_vals, r, c=color, s=20, alpha=0.7)
        plt.ylabel('order parameter (r)')
        plt.xlabel('coupling (K)')
        sns.despine()
    
    
    def plot_opxt(self):
        '''
        Instantiates a random graph and transforms it into an adjacency matrix,
        then plots the phase coherence chart for the Kuramoto model.
        Inputs:
            self
        Outputs:
            plot object with the Kuramoto model time series of all nodes vs time
        '''
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.6)
           
        graph = nx.to_numpy_array(self.G)
    
        # Creating model instance model with parameters
        model = Kuramoto(coupling=3, dt=0.01, T=10, n_nodes=len(graph))
    
        # Run simulation
        act_mat = model.run(adj_mat=graph)
        plot_phase_coherence(act_mat)
