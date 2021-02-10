#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import networkx as nx
import math
from sklearn.linear_model import Lasso

def demical(x):
    """
    Encode the input into demical with log transformation.
    
    Input: np.array or list, a vector with elements 0 and 1 (view as a binary).
    
    Output: np.array, size = 1.
    
    """
    l = x.size
    tmp = [x[i]*2**i for i in range(l)]
    return np.log(1+np.array(tmp).sum())

def binary(x, sup):
    """
    Decode the input into binary with exponential transformation.
    
    Input: x, np.array, a real number.
        
        sup, int, expected size of output. 
    
    Output: np.array, a vector with element 0 and 1 (view as a binary).
    """
    tmp = np.zeros(sup)
    x = round(np.exp(x)-1)
    tmp = [(x//2**i)%2 for i in range(sup)]
    return np.array(tmp)

def re_id(G):
    v = list(G.nodes())
    v.sort(reverse = False)
    reid = {}
    for node in G.nodes:
        reid[node] = v.index(node)
    tmp = nx.Graph()
    for (u,v) in G.edges:
        tmp.add_edge(reid[u], reid[v])
    return tmp
    
def phi(v, node):
    """
    Input: v, list.
        
        node, element to be located.
   
    Output: int, the index of node in list v.
    """
    if node not in v:
        return 0
    else:
        return v.index(node)
    
def encode(G, g):
    """
    Input: G, nx.Graph, 
        
        g, nx.Graph, the subgraph to be encoded.
    
    Output: np.array, adjacency vector of g.
    """
    node_set = list(g.nodes())
    node_set.sort(reverse = False)
    n = len(list(G.nodes()))
    k = len(node_set)
    ind = np.zeros(n)
    for node in node_set:
        tmp = np.zeros(k)
        for u in g.neighbors(node):
            tmp[phi(node_set, u)] = 1
        ind[node] = demical(tmp)
    return ind

def decode(x):
    """
    Input: x, np.array, adjacency vector of an unknown subgraph.
    
    Output: nx.Graph, the subgraph we reconstruct refer to the adjacency vector.
    """
    if len(x) == 0:
        return nx.Graph()
    ind = np.where(x>0)[0].tolist()
    g = nx.Graph()
    g.add_nodes_from(ind)
    for node in ind:
        tmp = binary(x[node], len(ind))
        tmp = np.array(ind)[tmp>0].tolist()
        if node in tmp:
            tmp.remove(node)
        g.add_edges_from([(node, x) for x in tmp])
    return g

def recover(y, Emb, alpha, eta, max_iter):
    """"
    Input: y, np.array, the embedding of an unknown subgraph.
        
        Emb, np.array, the matrix of node embeddings.
        
        alpha, float, the regularization parameter of Lasso regression.
        
        eta, float, the threshold for regularization.
        
        max_iter, int, the maximum iteration number of Lasso regression.
    
    Output: np.array, the solution of adjacency vector.
    """
    sol = list()
    lasso = Lasso(alpha = alpha, max_iter = max_iter)
    lasso.fit(Emb, y)
    index2 = np.where(lasso.coef_>eta)[0]
    index1 = np.zeros(index2.size)
    while np.array(index1 != index2).any():
        index1 = index2
        C = np.linalg.lstsq(Emb[:,index1], y, rcond=None)
        sol = np.zeros(Emb.shape[1])
        sol[index1]=C[0]
        index2 = np.where(sol>eta)[0]
    return sol


def test_decoder(g,G,Emb,alpha,eta, max_iter):
    """
    Input: g, nx.Graph, the subgraph to be tested.
        
        G, nx.Graph, the entire graph.
        
        alpha, float, the regularization parameter of Lasso regression.
        
        eta, float, the threshold in recover function.
        
        max_iter, int, maximum iteration number of Lasso regression.
        
    Output: test, nx.Graph, the subgraph reconstructed.
         
         error, float, error of the recovery of adjacency vector.
    """
    n = len(G.nodes)
    sub = encode(G,g)
    y = np.dot(Emb, sub)
    tmp = recover(y, Emb, alpha, eta, max_iter)
    test = decode(tmp)
    if test:
        error = np.linalg.norm(tmp-sub)
        return (test, error)
    else:
        return(g, -1)

