
import numpy as np
import networkx as nx
from numpy.linalg import pinv
from utils import *


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
    print('\n')
    print(most_correlated_nodes)
    #print(list_of_used_nodes.sort())
    print('\n')
    
    print(len(most_correlated_nodes))

    
    for k in range(len(most_correlated_nodes)):
       u=int(most_correlated_nodes[k][0])
       v=int(most_correlated_nodes[k][1])
       G = nx.contracted_nodes(G,u,v)
    print(G.number_of_nodes())
    G=nx.convert_node_labels_to_integers(G)
    return G         


def laplacian_renorm(G,number_of_steps):
    for j in range(number_of_steps):
           G=coarse_graining_step(G)
    return G

def laplacian_replica(G,Gr,error):
    L=nx.laplacian_matrix(G)
    L=L.todense()
    L=np.array(L)
    r_0=0.01
    r_0=0.0
    H=r_0+L
    C=np.linalg.pinv(H)
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
    correlated_nodes_sorted= correlated_nodes[x]
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