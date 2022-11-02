# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 20:43:45 2022

@author: Costelinha
"""
import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import powerlaw

from scipy.stats import norm,lognorm,cauchy
from numpy.linalg import eigh,eigvalsh,inv, norm, matrix_rank, pinv
from collections import defaultdict
from sklearn.linear_model import LinearRegression

from utils import *

"""
All functions take as input the graph G.
For the power-law distribution the input x is the starting point of the fitting.
"""

def plot_degree_dist(G):
    degrees = [G.degree(n) for n in G.nodes()]
    #print(degrees)
    
   # plt.xscale('log')
    data =degrees

    # Fit a normal distribution to the data:
    mu, std = norm.fit(data)

    # Plot the histogram.
    plt.hist(data, bins=25, density=True, alpha=0.6, color='g')

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)
    plt.xlabel('degrees')
    #plt.yscale('log')
    plt.show()
    
   

def plot_degree_dist_cauchy(G):
    
    
    
    degrees = [G.degree(n) for n in G.nodes()]
    #print(degrees)
    
   # plt.xscale('log')
    data =degrees

   
   # Fit a cauchy distribution to the data:
    mu, std = cauchy.fit(data)

    # Plot the histogram.
    plt.hist(data, bins=25, density=True, alpha=0.6, color='g')

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = cauchy.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)
    
    plt.xlabel('degrees')
    plt.show()
    
    #fit = powerlaw.Fit(degrees,xmin=x  ,discrete=True)
    #fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit pdf')
    #fit.plot_pdf( color= 'b')
    #plt.xlabel('k')
    #plt.ylabel('p(k)')
    #plt.show()
    #print('beta= ',fit.power_law.alpha,'  sigma= ',fit.power_law.sigma)


def plot_degree_dist_lognorm(G):
    
    
    
    degrees = [G.degree(n) for n in G.nodes()]
    #print(degrees)
    
   # plt.xscale('log')
    data =degrees

    # Fit a normal distribution to the data:
   
    mu, std,scale = lognorm.fit(data)

    # Plot the histogram.
    plt.hist(data, bins=25, density=True, alpha=0.6, color='g')

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = lognorm.pdf(x, mu, std, scale)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: s = %.4f,  loc = %.2f, scale=%.2f" % (mu, std,scale)
    plt.title(title)
    plt.xlabel('degrees')
    plt.show()
    
   
    
def laplacian_distrubution(G):
 
    #L=nx.laplacian_matrix(G)
    L=nx.normalized_laplacian_matrix(G)
    L=L.todense()
    L=np.array(L)
    
    l=eigvalsh(L)
    data =l

    # Fit a normal distribution to the data:
    mu, std = norm.fit(data)

    # Plot the histogram.
    plt.hist(data, bins=25, density=True, alpha=0.6, color='b')

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)
    plt.xlabel('Eigenvalues of Laplacian ')
    plt.ylabel('Number of eigenvalues')
    plt.show()
    
    
   # plt.xscale('log')
    #plt.yscale('log')
    
    #plt.xlim(xmin=0, xmax=20)
    
    
def laplacian_distrubution_cauchy(G):
    
    #L=nx.laplacian_matrix(G)
    L=nx.normalized_laplacian_matrix(G)
    L=L.todense()
    L=np.array(L)
    
    l=eigvalsh(L)
    data =l

    # Fit a normal distribution to the data:
    mu, std = cauchy.fit(data)

    # Plot the histogram.
    plt.hist(data, bins=25, density=True, alpha=0.6, color='b')

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = cauchy.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.xlim(left=0,right=2)
    plt.title(title)
    plt.xlabel('Eigenvalues of Laplacian ')
    plt.ylabel('Number of eigenvalues')
    plt.show()
    
    
   

def laplacian_distrubution_lognorm(G):
    
    #L=nx.laplacian_matrix(G)
    L=nx.normalized_laplacian_matrix(G)
    L=L.todense()
    L=np.array(L)
    
    l=eigvalsh(L)
    data =l

    # Fit a lognormal distribution to the data:
    mu, std,scale = lognorm.fit(data)
    
    # Plot the histogram.
    plt.hist(data, bins=25, density=True, alpha=0.6, color='b')

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = lognorm.pdf(x, mu, std, scale)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: s = %.4f,  loc = %.2f, scale=%.2f" % (mu, std,scale)
    plt.title(title)
    plt.xlim(left=0,right=2)
    plt.xlabel('Eigenvalues of Laplacian ')
    plt.ylabel('Eigenvalue distribution')
    plt.show()
  
      
def clustering_distribution(G):
   
    clustering=nx.clustering(G)
    data=np.array(list(clustering.values()))
     # Fit a normal distribution to the data:
    mu, std = norm.fit(data)

    # Plot the histogram.
    plt.hist(data, bins=25, density=True, alpha=0.6, color='r')

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)
    plt.xlabel('clustering coeficients ')
    plt.ylabel('Number of clustering coeficients')
    plt.show()
  
    
def clustering_distribution_cauchy(G):
    
    clustering=nx.clustering(G)
    data=np.array(list(clustering.values()))
     # Fit a cauchy distribution to the data:
    mu, std = cauchy.fit(data)

    # Plot the histogram.
    plt.hist(data, bins=25, density=True, alpha=0.6, color='r')

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = cauchy.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)  
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)
    plt.xlabel('clustering coeficients ')
    plt.ylabel('Number of clustering coeficients')
    plt.show()
  
    
def clustering_distribution_lognorm(G):
   
    clustering=nx.clustering(G)
    data=np.array(list(clustering.values()))
     # Fit a lognormal distribution to the data:
    mu, std,scale = lognorm.fit(data)

    # Plot the histogram.
    plt.hist(data, bins=25, density=True, alpha=0.6, color='r')

    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = lognorm.pdf(x, mu, std, scale)
    plt.plot(x, p, 'k', linewidth=2)  
    title = "Fit results: s = %.2f,  loc = %.2f, scale=%.2f" % (mu, std,scale)
    plt.title(title)
    plt.xlabel('clustering coeficients ')
    plt.ylabel('Number of clustering coeficients')
    plt.show()
    
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
    heights,bins=np.histogram(l,bins=100)
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
    plt.scatter(l,cpd,label=lb)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('c(k/<k>)')
    plt.xlabel('k/<k>')
    #plt.ylim(bottom=10**(-3),top=1)
    plt.rcParams.update({'font.size': 15})
    plt.legend()
    
    plt.xlim(left=10**(-2),right=10**2)


def plot_nu_regression(G0,G1,G2,G3,G4,G5):
    
    kmean=[Average_degree(G0),Average_degree(G1),Average_degree(G2),Average_degree(G3),Average_degree(G4),Average_degree(G5)]
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
