#!/usr/bin/env python
# coding: utf-8

# Evaluate the performance of subgraph embedding on graph reconstruction task.

import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import Lasso
from RSE import encode, decode, test_decoder, re_id
import random
import argparse

def Reconstruction(G, Emb, node_set, alpha, mu, max_iter):
    err = list()
    Rec = {}
    Rec_G = nx.Graph()
    for node in node_set:
        nbunch = [u for u in G.neighbors(node)]
        if len(nbunch) > 20:
            random.shuffle(nbunch)
            nbunch = nbunch[:20]
        nbunch.append(node)
        g = nx.subgraph(G,nbunch)
        Rec[node] = np.dot(Emb, encode(G,g))
        (Rec_g,error) = test_decoder(g,G,Emb,alpha,mu, max_iter)
        err.append(error)
        for (u,v) in Rec_g.edges:
            Rec_G.add_edge(u,v)
    return Rec_G

def compute_error(G, Rec_G):
    for node in G.nodes:
        Rec_G.add_node(node)
    n = len(G.nodes)
    A = np.zeros((n,n))
    for node in G.nodes:
        tmp = [x for x in G[node]]
        A[node][tmp] = 1
    B = np.zeros((n,n))
    for node in Rec_G.nodes:
        tmp = [x for x in Rec_G[node]]
        B[node][tmp] = 1
    return np.count_nonzero(A-B)/2



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process filename.')
    parser.add_argument('--graphfile', type=str, help='filename of graph')
    parser.add_argument('--embeddingfile', type=str, help='filename of graph embedding')
    parser.add_argument('--alpha', type=float, help='regularization parameter')
    parser.add_argument('--mu', type=float, help='threshold of decoding')
    parser.add_argument('--max_iter', type=int, help='max iteration of decoding')
    args = parser.parse_args()
    
    graphfile = args.graphfile
    embeddingfile = args.embeddingfile
    alpha = args.alpha
    mu = args.mu
    max_iter = args.max_iter

    G = nx.read_edgelist(graphfile, nodetype=int)
    Emb = np.loadtxt(embeddingfile)
    G.remove_edges_from(nx.selfloop_edges(G))
    
    node_set = set(G.nodes())
    Rec = Reconstruction(G, Emb, node_set, alpha, mu, max_iter)
    error = compute_error(G, Rec)
    print("The number of error edges:", error)








