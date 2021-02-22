#!/usr/bin/env python
# coding: utf-8

import os
import time
import dgl
import torch
import numpy as np
import networkx as nx
from sklearn.linear_model import Lasso
from RSE import encode, decode, test_decoder
from util import load_data, separate_data
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from torch.autograd import Function
from torch.autograd import Variable


def DCT(n):
    tmp = np.array(range(n))
    tmp = tmp*2*n/np.pi
    tmp = [tmp*(2*x+1) for x in range(n)]
    tmp = [np.cos(x) for x in tmp]
    scale = np.ones(n) * np.sqrt(2)
    scale[0] = 1
    scale = np.diag(scale)/np.sqrt(n)
    return np.dot(np.array(tmp), scale)

def load_dataset(dataset):
    data = load_data(dataset,True)
    return data
    
def preprocessing(data, emb_file, seed, trans):
    num_graphs = len(data[0])
    nx_graphs = [data[0][i].g for i in range(num_graphs)]
    dgl_graphs = [dgl.from_networkx(graph) for graph in nx_graphs]
    batch_graphs = dgl.batch(dgl_graphs)
    num_nodes = len(batch_graphs.nodes())
    graph_size = [len(g.nodes()) for g in nx_graphs]
    
    emb = np.loadtxt(emb_file)
    if trans:
        emb = np.dot(emb,DCT(num_nodes).T)
    G = batch_graphs.to_networkx()
    Sub = {}
    for i in range(num_graphs):
        if i == 0:
            node_start = 0
        else:
            node_start = sum(graph_size[:i-1]) 
        node_end = sum(graph_size[:i])
        nbunch = [node for node in range(node_start, node_end)]
        subgraph = nx.subgraph(G, nbunch)
        Sub[data[0][i]] = np.dot(emb, encode(G,subgraph))
    
    idx_list = separate_data(data[0],seed = seed)
    return Sub, idx_list, data[0]

def training_data(data, Sub, id_list, fold_idx):
    
    train_idx, test_idx = idx_list[fold_idx]
    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]
    
    
    train_emb = torch.Tensor([Sub[graph] for graph in train_graph_list])
    train_label = torch.Tensor([graph.label for graph in train_graph_list]).long()
    train_set = TensorDataset(train_emb, train_label)
    train_loader = DataLoader(train_set, shuffle=False, batch_size = 10)
    
    
    test_emb = torch.Tensor([Sub[graph] for graph in test_graph_list])
    test_label = torch.Tensor([graph.label for graph in test_graph_list]).long()
    test_set = TensorDataset(test_emb, test_label)
    test_loader = DataLoader(test_set, shuffle=False)
    
    return train_loader, test_loader


class MLP(torch.nn.Module):
    def __init__(self, p, n_labels):
        super(MLP,self).__init__()
        self.fc1 = torch.nn.Linear(p,256)
        self.fc2 = torch.nn.Linear(256,256)
        self.fc3 = torch.nn.Linear(256,n_labels)
        
    def forward(self,din):
        dout = torch.tanh(self.fc1(din))
        dout = torch.tanh(self.fc2(dout))
        return torch.sigmoid(self.fc3(dout))


def train(train_loader, test_loader, max_iter, lr):
    model = MLP(256 ,3).cuda()
    opt = torch.optim.Adam(model.parameters(),lr=lr)
    lossfunc = nn.CrossEntropyLoss()

    for epoch in range(max_iter):
        train_loss = []
        if epoch:
            tmp = np.array(acc).mean()
        acc = []
        for i,data in enumerate(train_loader):
    
            opt.zero_grad()
            (emb, label) = data
            emb = torch.autograd.Variable(emb).cuda()
            label = torch.autograd.Variable(label).cuda()
            pred = model(emb)
    
            loss = lossfunc(pred, label)
            train_loss.append(loss.item())
            loss.backward()
            opt.step()
    
        for i,data in enumerate(test_loader):
            (emb, label) = data
            emb = torch.autograd.Variable(emb).cuda()
            pred = model(emb)
            pred = pred.argmax()
            acc.append(int(pred)==int(label))
            
    
        if epoch%5 == 0:
            print(epoch,",train loss:",np.array(train_loss).mean(),",test error:", np.array(acc).mean())
    
    return np.array(acc).mean()

if __name__= '__main__':
    parser = argparse.ArgumentParser(description='Process filename.')
    parser.add_argument('--dataset', type=str, help='name of subgrah classification dataset')
    parser.add_argument('--embeddingfile', type=str, help='filename of graph embedding')
    parser.add_argument('--trans', type=bool, help='whether do DCT or not')
    parser.add_argument('--max_iter', type=int, help='max iteration of training')
    parser.add_argument('--lr', type=float, help='learning rate of training')
    args = parser.parse_args()
    
    dataset = args.dataset
    emb_file = args.embeddingfile
    trans = args.trans
    max_iter = args.max_iter
    lr = args.lr
    
    data = load_dataset(dataset)
    err = []
    Sub, idx_list, graph_list = preprocessing(data, emb_file, None, trans)
    for i in range(10):
        train_loader, test_loader = training_data(data, Sub, idx_list, i)
        print("Idx:",i)
        err.append(train(train_loader, test_loader, max_iter, lr))
    print("Average of accuracy:",np.array(err).mean())
    print("Standard deviation of accuracy:",np.array(err).std())




