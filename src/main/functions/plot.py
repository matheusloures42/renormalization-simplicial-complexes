# -*- coding: utf-8 -*-
"""
@author: Costelinha
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import powerlaw

from scipy.stats import norm, lognorm, cauchy
from numpy.linalg import eigvalsh, norm
from collections import defaultdict
from sklearn.linear_model import LinearRegression

from utils import *

"""
All functions take as input the graph G.
For the power-law distribution the input x is the starting point of the fitting.
"""

def plot_distribution(
    G, plot_type="degree", fit_type="normal", x_resolution=100, x_log=False, y_log=False
):
    '''
    Plots the degree distribution of a graph
    Inputs:
        G: networkX graph object
        plot_type: string stating the type of plot you want. 
            Options: "degree", "eigenvalue", "clustering coefficient"
            Default: "degree"
        fit_type: string stating the type of fit you want. 
            Options: "normal", "cauchy", "lognormal"
            Default: "normal"
        x_resolution: int representing the number of points in the x-axis
            Default: 100
        x_log: bool that states if the chart is logarithmic in the x-axis
            Default: False
        y_log: bool that states if the chart is logarithmic in the y-axis
            Default: False
    Outputs:
        plot object with the degree distribution of the graph
    '''
    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')

    x_label = plot_type.capitalize() + 's'

    # Computing the y values depending on plot type
    if plot_type == "degree":
        data = [G.degree(n) for n in G.nodes()]
    elif plot_type == "eigenvalue":
        L_matrix = nx.normalized_laplacian_matrix(G)
        L_array = np.array(L_matrix.todense())
        data = eigvalsh(L_array)
    elif plot_type == "clustering coefficient":
        clustering = nx.clustering(G)
        data = np.array(list(clustering.values()))

    # Setting up x-axis
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, x_resolution)

    # Getting the desired type of fit
    if fit_type == "normal":
        mu, std = norm.fit(data)
        p = norm.pdf(x, mu, std)
    elif fit_type == "cauchy":
        mu, std = cauchy.fit(data)
        p = cauchy.pdf(x, mu, std)
    elif fit_type == "lognormal":
        mu, std, scale = lognorm.fit(data)
        p = lognorm.pdf(x, mu, std, scale)

    # Plotting the histogram
    plt.hist(data, bins=25, density=True, alpha=0.6, color='g')
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel('Count')
    plt.show()
    
   
    
# def laplacian_distribution(G):
 
#     #L=nx.laplacian_matrix(G)
#     L=nx.normalized_laplacian_matrix(G)
#     L=L.todense()
#     L=np.array(L)
    
#     l=eigvalsh(L)
#     data =l

#     # Fit a normal distribution to the data:
#     mu, std = norm.fit(data)

#     # Plot the histogram.
#     plt.hist(data, bins=25, density=True, alpha=0.6, color='b')

#     # Plot the PDF.
#     xmin, xmax = plt.xlim()
#     x = np.linspace(xmin, xmax, 100)
#     p = norm.pdf(x, mu, std)
#     plt.plot(x, p, 'k', linewidth=2)
#     title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
#     plt.title(title)
#     plt.xlabel('Eigenvalues of Laplacian ')
#     plt.ylabel('Number of eigenvalues')
#     plt.show()
  
      
# def clustering_distribution(G):
   
#     clustering=nx.clustering(G)
#     data=np.array(list(clustering.values()))
#      # Fit a normal distribution to the data:
#     mu, std = norm.fit(data)

#     # Plot the histogram.
#     plt.hist(data, bins=25, density=True, alpha=0.6, color='r')

#     # Plot the PDF.
#     xmin, xmax = plt.xlim()
#     x = np.linspace(xmin, xmax, 100)
#     p = norm.pdf(x, mu, std)
#     plt.plot(x, p, 'k', linewidth=2)
#     title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
#     plt.title(title)
#     plt.xlabel('clustering coeficients ')
#     plt.ylabel('Number of clustering coeficients')
#     plt.show()
  
    
    


    #Here x
def plot_degree_dist_power_law(G,x):
   
    degrees = [G.degree(n) for n in G.nodes()]
    l=degrees

    # Plot the histogram.
    plt.hist(l, bins=50, density=True, alpha=0.6, color='g')

    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlim(xmin=0)
    fit = powerlaw.Fit(np.array(l),xmin=x,discrete=True)
    
    fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit pdf')
    fit.plot_pdf( color= 'b')
    
    plt.ylabel('degree distribution')
    plt.xlabel('degrees')
    plt.show()
    print('gama= ',fit.power_law.alpha,'  sigma= ',fit.power_law.sigma)
    plt.rcParams.update({'font.size': 22})
    

def plot_all_dist(G):
    
    degrees = [G.degree(n) for n in G.nodes()]
    kmean=Average_degree(G)
    l=degrees/kmean
    heights, _=np.histogram(l,bins=100)
    pk=heights/np.sum(heights)
    x=np.linspace(0, np.amax(l),num=len(heights))
    plt.scatter(x,pk)
    plt.plot(x,pk)
    plt.xscale('log')
    plt.yscale('log')
    plt.rcParams.update({'font.size': 22})
    
    plt.ylabel('p(k/<k>)')
    plt.xlabel('k/<k>')
    

def plot_ccdf(G,lb):
    
    degrees = [G.degree(n) for n in G.nodes()]
    kmean=Average_degree(G)
    l=degrees/kmean
    heights,bins=np.histogram(l,bins=200)
    pk=heights/np.sum(heights)
    x=np.linspace(0, np.amax(l),num=len(heights))
    ccdf=1-np.cumsum(pk)
    plt.scatter(x,ccdf,label=lb)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(bottom=10**(-4),top=1)
    plt.rcParams.update({'font.size': 22})
    plt.legend()
    plt.xlim(left=10**(-2),right=10**3)
    plt.ylabel('p(k/<k>)')
    plt.xlabel('k/<k>')
    

def clustering_per_kl(G,lb):
    
    degrees = [G.degree(n) for n in G.nodes()]
    kmean=Average_degree(G)
    s=[]
    for i in degrees:
        if i not in s:
            s.append(i)
    l=s/kmean
   
    d = defaultdict(list)
    K=0
    for u in G.nodes():
       d[G.degree(u)].append(u)
    cpd=[]
    for degree in d:
        K+=1
        clustering_coeff = nx.clustering(G, d[degree])
        cpd.append(sum(clustering_coeff.values())/len(clustering_coeff))

    plt.scatter(l,cpd/kmean,label=lb)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('c(k/<k>)')
    plt.xlabel('k/<k>')
    #plt.ylim(bottom=10**(-3),top=1)
    plt.rcParams.update({'font.size': 15})
    plt.legend()
    
    plt.xlim(left=10**(-2),right=10**2)


def plot_nu_regression(G0,G1,G2,G3,G4,G5):

    G_list=[G0,G1,G2,G3,G4,G5]
    kmean=[Average_degree(g) for g in G_list]
    l=np.arange(6)

    plt.scatter(l,np.log(kmean))
    regressor = LinearRegression() 
    regressor.fit(np.array(l).reshape((-1,1)),np.array(np.log(kmean)) )
    ypred=regressor.predict(np.array(l).reshape((-1,1)) )
    plt.plot(l,ypred)
    plt.ylabel('log(<k>)')
    plt.xlabel('layers')
    print('coef linear=',regressor.intercept_)
    print('coef angular=',regressor.coef_)


def plot_nu_regression2(G0,G1,G2):
   
    kmean=[Average_degree(G0),Average_degree(G1),Average_degree(G2)]
    l=np.arange(3)
    plt.scatter(l,np.log(kmean))
    regressor = LinearRegression() 
    regressor.fit(np.array(l).reshape((-1,1)),np.array(np.log(kmean)) )
    ypred=regressor.predict(np.array(l).reshape((-1,1)) )
    plt.plot(l,ypred)
    plt.ylabel('log(<k>)')
    plt.xlabel('layers')
    plt.show()
    print('coef linear=',regressor.intercept_)
    print('coef angular=',regressor.coef_)


def eigenvector_centrality_x_kl(G,lb):
    degrees = [G.degree(n) for n in G.nodes()]
    kmean=Average_degree(G)
    s=[]
    for i in degrees:
        if i not in s:
            s.append(i)
    l=s/kmean
    
    d = defaultdict(list)
    K=0
    for u in G.nodes():
       d[G.degree(u)].append(u)
    cpd=[]
    
    centrality_coeff = nx.eigenvector_centrality(G)
    values= list(centrality_coeff.values())
    
    for degree in d:
        sum=0
        for index in d[degree]:
            sum+=values[index]
        cpd.append(sum/len(d[degree]))

    plt.scatter(l,cpd,label=lb)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('eigenvector centrality')
    plt.xlabel('k/<k>')  
    plt.legend()


def plot_ccdf_weighted(G,lb):
    
    degrees = [G.degree(n,weight='weight') for n in G.nodes()]
    kmean=Average_degree_weighted(G)
    l=degrees/kmean
    heights,bins=np.histogram(l,bins=200)
    pk=heights/np.sum(heights)
    x=np.linspace(0, np.amax(l),num=len(heights))
    cdf=1-np.cumsum(pk)
    plt.scatter(x,cdf,label=lb)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(bottom=10**(-4),top=1)
    plt.legend()
    plt.xlim(left=10**(-2),right=10**3)
    plt.ylabel('ccdf')
    plt.xlabel('s/<s>')
    

def clustering_per_sl(G,lb):
    
    degrees = [G.degree(n,weight='weight') for n in G.nodes()]
    kmean=Average_degree_weighted(G)
    s=[]
    for i in degrees:
        if i not in s:
            s.append(i)
    l=s/kmean

    d = defaultdict(list)
    K=0
    for u in G.nodes():
       d[G.degree(u,weight='weight')].append(u)
    cpd=[]
    for degree in d:
        K+=1
        clustering_coeff = nx.clustering(G, d[degree],weight='weight')
        
        cpd.append(sum(clustering_coeff.values())/len(clustering_coeff))
    plt.scatter(l,cpd,label=lb)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('c(s/<s>)')
    plt.xlabel('s/<s>')
    #plt.ylim(bottom=10**(-3),top=1)
    plt.rcParams.update({'font.size': 12})
    plt.legend()
    
    plt.xlim(left=10**(-2),right=10**2)


def eigenvector_centrality_x_sl(G,lb):
    degrees = [G.degree(n,weight='weight') for n in G.nodes()]
    kmean=Average_degree_weighted(G)
    s=[]
    for i in degrees:
        if i not in s:
            s.append(i)
    l=s/kmean
    
    d = defaultdict(list)
    K=0
    for u in G.nodes():
       d[G.degree(u,weight='weight')].append(u)
    cpd=[]
    
    centrality_coeff = nx.eigenvector_centrality(G,weight='weight')
    values= list(centrality_coeff.values())
    
    for degree in d:
        sum=0
        for index in d[degree]:
            sum+=values[index]

        cpd.append(sum/len(d[degree]))

    plt.scatter(l,cpd,label=lb)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('eigenvector centrality')
    plt.xlabel('s/<s>')  
    plt.legend()


def k_x_s(G,lb):

    degreesw = [G.degree(n,weight='weight') for n in G.nodes()]
    degrees = [G.degree(n) for n in G.nodes()]
    d = defaultdict(list)
    k=[]
    for i in degrees:
        if i not in k:
            k.append(i)
   
    for u in G.nodes():
        d[G.degree(u)].append(u)
    sk=[]
    for degree in d:
        sum=0
        for index in d[degree]:
            sum+=degreesw[index]
        sk.append(sum/len(d[degree]))
    
    regressor = LinearRegression() 
    regressor.fit(np.array(k).reshape((-1,1)),np.array(sk) )
    ypred=regressor.predict(np.array(k).reshape((-1,1)) )
    plt.scatter(k,sk,label=lb)
    plt.legend()
    plt.plot(k,ypred)
    plt.ylabel('s')
    plt.xlabel('k')
    print('coef linear=',regressor.intercept_)
    print('coef angular=',regressor.coef_)
    

def plot_clustering_x_l(G,l):
    c=nx.average_clustering(G)
    plt.scatter(c,l,color='b')


def plot_avg_path_length_x_l(G,l):
    L=nx.average_shortest_path_length(G)
    plt.scatter(L,l, color='red')


def average_neighbor_degree_x_kl(G,lb):
    degrees = [G.degree(n) for n in G.nodes()]
    kmean=Average_degree(G)
    ksqrmean=average_degree_square(G)
    s=[]
    for i in degrees:
        if i not in s:
            s.append(i)
    l=s/kmean
    
    d = defaultdict(list)
    K=0
    for u in G.nodes():
       d[G.degree(u)].append(u)
    annd=[]
    for degree in d:
        K+=1
        nearest_neighbors_degree = nx.average_neighbor_degree(G, nodes=d[degree])
        annd.append(
            sum(
                nearest_neighbors_degree.values()
            ) / len(
                nearest_neighbors_degree
            )*(kmean/ksqrmean)
        )
    
    plt.scatter(l,annd,label=lb)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(bottom=10**(-1),top=10)
    plt.ylabel('knn,n(k/<k>)')
    plt.xlabel('k/<k>') 