#!/usr/bin/env python
# coding: utf-8

import os
import time
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import Lasso
from dgl.data import citation_graph as citegrh
from RSE import encode, decode, test_decoder, re_id
import random 
import matplotlib.pyplot as plt

def random_walk(G, k):
    Nodes = [x for x in G.nodes()]
    node = random.sample(Nodes, 1)[0]
    while not [u for u in G[node]]:
        node = random.sample(Nodes, 1)[0]
    g = nx.Graph()
    g.add_node(node)
    tStart = time.time()
    while len(g.nodes)<k:
        neighbors = [u for u in G[node]]
        node_next = random.sample(neighbors, 1)[0]
        if (node, node_next) not in g.edges():
            g.add_edge(node, node_next)
        node = node_next
        tCurrent = time.time()
        if (tCurrent - tStart) > 1:
            break
    return g

def test_reversible(G, Emb, k, alpha, mu, induced):
    acc = list()
    g_list = list()
    for i in range(100):
        g = random_walk(G, k)
        if len(g.nodes) < k:
            g = random_walk(G,k)
        g.remove_edges_from(nx.selfloop_edges(g))
        g_list.append(g)
        (Rec_g,error) = test_decoder(g,G,Emb,alpha, mu, 10000)
        if error == -1:
            acc.append(0)
            continue
        if g.edges() == Rec_g.edges():
            acc.append(1)
        else:
            acc.append(0)
    return acc,g_list


def plot(filename):
    recover_accuracy = np.load(filename, allow_pickle=True).item()
    x = [k for k in recover_accuracy]
    x.sort()
    y = [recover_accuracy[k] for k in x]
    x = np.array(x)
    y = np.array(y)
    plt.plot(x,y)
    
    
if __name__= '__main__':
    parser = argparse.ArgumentParser(description='Process filename.')
    parser.add_argument('--graphfile', type=str, help='filename of graph data')
    parser.add_argument('--embeddingfile', type=str, help='filename of graph embedding')
    parser.add_argument('--alpha', type=float, help='regularization parameter of decoding')
    parser.add_argument('--mu', type=float, help='threshold in decoding algorithm')
    parser.add_argument('--induced', type=bool, help='induced graph or embedded graph')
    parser.add_argument('--accfile', type=str, help='filename of recovery accuracy')
    args = parser.parse_args()

    graphfile = args.graphfile
    G = nx.read_adjlist(graphfile,nodetype=int)
    G.remove_edges_from(nx.selfloop_edges(G))

    filename = args.embeddingfile
    Emb = np.loadtxt(filename)

    alpha = args.alpha
    mu = args.mu
    induced = args.induced
    recover_accuracy = {}

    for k in range(2,40): 
        tStart = time.time()
        acc, g_list = test_reversible(G, Emb, k, alpha, mu, induced)
        tEnd = time.time()
        recover_accuracy[k]= sum(acc)/len(acc)
        print("Time used:", tEnd - tStart, "Size:",k,"Accuracy:", recover_accuracy[k])

    filename = args.accfile
    np.save(filename, recover_accuracy)
    plot(filename)
