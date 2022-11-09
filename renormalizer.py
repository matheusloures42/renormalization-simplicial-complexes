
import numpy as np
import networkx as nx
from numpy.linalg import pinv
from utils import *


"""
coarse graining step use G as unput and make one step of renormalization using the pseudo inverse of laplacian.

"""

def coarse_graining_step(G):
    L=nx.laplacian_matrix(G)
    L=L.todense()
    L=np.array(L)
    r_0=0.01
    r_0=0.0
    H=r_0+L
    C=np.linalg.pinv(H)
    most_correlated_nodes=[]
    correlated_nodes=[]
    correlations=[]
    for i in range(len(C)):
        for j in range(len(C)):
            if i>j:
                correlated_nodes.append([i,j])
                correlations.append(C[i,j])
                
    correlations=np.array(correlations)          
    x=correlations.argsort()
    correlated_nodes=np.array(correlated_nodes)
    correlated_nodes_sorted= correlated_nodes[x[::-1]]
    
    list_of_used_nodes=[]
    for i in range(len(x)):
        for j in range(2):
            if i==0:
                most_correlated_nodes.append(correlated_nodes_sorted[i].tolist())
                list_of_used_nodes.append(int(correlated_nodes_sorted[i][j]))
               
            if int(correlated_nodes_sorted[i][j]) not in list_of_used_nodes:
                if j==0:
                    if int(correlated_nodes_sorted[i][j+1]) not in list_of_used_nodes:
                        most_correlated_nodes.append(correlated_nodes_sorted[i].tolist())
                        list_of_used_nodes.append(correlated_nodes_sorted[i][j])
                        list_of_used_nodes.append(correlated_nodes_sorted[i][j+1])
                        
                if j==1:
                    if int(correlated_nodes_sorted[i][j-1]) not in list_of_used_nodes:
                        most_correlated_nodes.append(correlated_nodes_sorted[i].tolist())
                        list_of_used_nodes.append(int(correlated_nodes_sorted[i][j]))
                        list_of_used_nodes.append(int(correlated_nodes_sorted[i][j-1]))        
    most_correlated_nodes.remove(most_correlated_nodes[0])    
    #print('\n')
    #print(most_correlated_nodes)
    #print(list_of_used_nodes.sort())
    #print('\n')
    
    #print(len(most_correlated_nodes))

    
    for k in range(len(most_correlated_nodes)):
       u=int(most_correlated_nodes[k][0])
       v=int(most_correlated_nodes[k][1])
       G = nx.contracted_nodes(G,u,v)
    #print(G.number_of_nodes())
    G=nx.convert_node_labels_to_integers(G)
    return G         

"""
Laplacian renorm uses G and the number of steps of renormalization that you want as input.
"""
def laplacian_renorm(G,number_of_steps):
    for j in range(number_of_steps):
           G=coarse_graining_step(G)
    return G

"""
tentative of replica,it takes the graph G, the renormalized graph GR and an error. We remove the most correlated links
of Gr until it have the  average degree in a neighborhood of the average degree of G defined by the given error.
"""

def laplacian_replica(G,Gr,error):
    L=nx.laplacian_matrix(G)
    L=L.todense()
    L=np.array(L)
    r_0=0.01
    r_0=0.0
    H=r_0+L
    C=np.linalg.pinv(H)
    N=len(C)
    correlated_nodes=[]
    correlations=[]
    for i in range(len(C)):
        for j in range(len(C)):
            if i>j:
                correlated_nodes.append([i,j])
                correlations.append(C[i,j])
    correlations=np.array(correlations)          
    x=correlations.argsort()
    correlated_nodes=np.array(correlated_nodes)
    correlated_nodes_sorted= correlated_nodes[x[::-1]]
    degrees = [G.degree(n) for n in G.nodes()]
    kmean=Average_degree(G)
    k=Average_degree(Gr)
    ind=0
    Gaux2=Gr.copy()
    Gaux2=nx.convert_node_labels_to_integers(Gaux2)
    l=0
    while k>kmean+error or k<kmean-error:
                
        u=int(correlated_nodes_sorted[l][0])
        v=int(correlated_nodes_sorted[l][1])
        
        if Gaux2.has_edge(u,v)==True:
            Gaux2.remove_edge(v,u)
            k=Average_degree(Gaux2)
            
            
        l+=1
    return Gaux2

def average_gamma_barabasi_renorm(N,m,number_of_simulations, number_of_steps, starting_points):
    gammas_list=[]
    error_list=[]
    gamma_renorm_list = [[] for _ in range(number_of_steps)]
    gamma_average_list=[]
    
    for i in range(number_of_simulations):
        if i>0:
            G.clear()
            K.clear()
        G=nx.barabasi_albert_graph(N,m)
        K = {}
        for j in range(number_of_steps):
            K[j]= laplacian_renorm(G,j+1)
            gamma_renorm_list[j].append(calculate_gamma(K[j],starting_points[j+1]))
  
        degrees = [G.degree(n) for n in G.nodes()]
        l=degrees
        fit = powerlaw.Fit(np.array(l),xmin=starting_points[0],discrete=True)
        gammas_list.append(fit.power_law.alpha)
    gamma_mean=np.mean(gammas_list)
    gamma_average_list.append(gamma_mean)
    
    for j in range(number_of_steps):
        gamma_average_list.append(np.mean(gamma_renorm_list[j]))
    return gamma_average_list
    
"""
functions for weighted graphs
"""

def weighted_laplacian_renorm(G,number_of_steps):
    for j in range(number_of_steps):
           G=weighted_coarse_graining_step(G)
    return G

def weighted_coarse_graining_step(G):
    L=nx.laplacian_matrix(G)
    L=L.todense()
    L=np.array(L)
    r_0=0.01
    r_0=0.0
    H=r_0+L
    C=np.linalg.pinv(H)
    most_correlated_nodes=[]
    correlated_nodes=[]
    correlations=[]
    #weights=list(list(G.edges(data=True))[0][-1].keys())
    for i in range(len(C)):
        for j in range(len(C)):
            if i>j:
                correlated_nodes.append([i,j])
                correlations.append(C[i,j])
                
    correlations=np.array(correlations)          
    x=correlations.argsort()
    correlated_nodes=np.array(correlated_nodes)
    correlated_nodes_sorted= correlated_nodes[x[::-1]]
    #weights_sorted=weights[x[::-1]]
    
    list_of_used_nodes=[]
    for i in range(len(x)):
        for j in range(2):
            if i==0:
                most_correlated_nodes.append(correlated_nodes_sorted[i].tolist())
                list_of_used_nodes.append(int(correlated_nodes_sorted[i][j]))
               
            if int(correlated_nodes_sorted[i][j]) not in list_of_used_nodes:
                if j==0:
                    if int(correlated_nodes_sorted[i][j+1]) not in list_of_used_nodes:
                        most_correlated_nodes.append(correlated_nodes_sorted[i].tolist())
                        list_of_used_nodes.append(correlated_nodes_sorted[i][j])
                        list_of_used_nodes.append(correlated_nodes_sorted[i][j+1])
                        
                if j==1:
                    if int(correlated_nodes_sorted[i][j-1]) not in list_of_used_nodes:
                        most_correlated_nodes.append(correlated_nodes_sorted[i].tolist())
                        list_of_used_nodes.append(int(correlated_nodes_sorted[i][j]))
                        list_of_used_nodes.append(int(correlated_nodes_sorted[i][j-1]))        
    most_correlated_nodes.remove(most_correlated_nodes[0])    
    #print('\n')
    #print(most_correlated_nodes)
    #print(list_of_used_nodes.sort())
    #print('\n')
    
    print(len(most_correlated_nodes))
    
    #for k in range(len(correlated_nodes)):
    Gcopy=G.copy()
    
    
    for k in range(len(most_correlated_nodes)):
        u=int(most_correlated_nodes[k][0])
        v=int(most_correlated_nodes[k][1])
        G = nx.contracted_nodes(G,u,v)
        for j in Gcopy.neighbors(v):
            if G.has_edge(u,j)==True:
                G[u][j]['weight']+= float(Gcopy[v][j]['weight'])
        
    G=nx.convert_node_labels_to_integers(G)
    #print(G.edges().data())
    return G  