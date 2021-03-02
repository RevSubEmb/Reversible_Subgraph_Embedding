#!/usr/bin/env python
# coding: utf-8

import torch
import dgl
import time
import numpy as np
import networkx as nx
import os
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
from dgl.data import citation_graph as citegrh
from dgl.nn import GraphConv
from sklearn.linear_model import Lasso
from util import load_data, separate_data

def DCT(n):
    tmp = np.array(range(n))
    tmp = tmp*2*n/np.pi
    tmp = [tmp*(2*x+1) for x in range(n)]
    tmp = [np.cos(x) for x in tmp]
    scale = np.ones(n) * np.sqrt(2)
    scale[0] = 1
    scale = np.diag(scale)/np.sqrt(n)
    return np.dot(np.array(tmp), scale)

        
class Net(nn.Module):
    def __init__(self, num_feature, p):
        super(Net, self).__init__()
        self.layer1 = GraphConv(num_feature, p)
        self.layer2 = GraphConv(p, p)
        self.layer3 = GraphConv(p, p)

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = F.relu(self.layer2(g, x))
        x = F.relu(self.layer3(g, x))
        return x

class NewLoss(nn.Module):
    def __init__(self):
        super(NewLoss, self).__init__()
        
    def forward(self, input_tensor, H):
        n, p = input_tensor.size()
        Psi = torch.tensor(DCT(n),dtype=torch.float32).cuda()
        Phi = torch.mm(Psi,input_tensor)
        result = torch.mm(Phi, Phi.t()) - H
        return torch.norm(result, 'fro')
    
    def backward(self, grad_output):
        return grad_output


def load_cora_data():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    g = dgl.from_networkx(data.graph)
    return g, features, labels

def compute_H(phi, mu, device):
    n, p = phi.size()
    psi = torch.tensor(DCT(n), dtype=torch.float32).to(device)
    tmp = torch.mm(psi, phi)
    gram = torch.mm(tmp, tmp.t())
    gram[gram>mu] = mu
    gram[gram<-mu] = -mu
    gram[range(n),range(n)] = 1
    return gram

def load_batch_graph(dataset):
    data = load_data(dataset,True)
    num_graphs = len(data[0])
    nx_graphs = [data[0][i].g for i in range(num_graphs)]
    dgl_graphs = [dgl.from_networkx(graph) for graph in nx_graphs]
    batch_graphs = dgl.batch(dgl_graphs)
    
    node_features = [data[0][i].node_features for i in range(num_graphs)]
    batch_features = torch.cat(node_features,0)
    graph_size = [len(g.nodes()) for g in nx_graphs]
    
    return batch_graphs, batch_features, graph_size


if __name__ = '__main__':
    parser = argparse.ArgumentParser(description='Process filename.')
    parser.add_argument('--dataset', type=str, help='name of grah dataset')
    parser.add_argument('--embeddingfile', type=str, help='filename to store graph embedding')
    parser.add_argument('--PATH', type=str, help='filename to store graph neural network')
    parser.add_argument('--p', type=int, help='dimension of embedding')
    parser.add_argument('--max_iter', type=int, help='max iteration of training')
    parser.add_argument('--lr', type=float, help='learning rate of training')
    parser.add_argument('--mu', type=float, help='threshold in computing H')
    args = parser.parse_args()
    
    device = ''
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    batch_graphs, batch_features, graph_size = load_batch_graph(args.dataset)
    
    g = batch_graphs.to(device)
    features = batch_features.to(device)

    net = Net(features.shape[1],args.p).to(device)
    lossfunc = NewLoss()

    dur = []

    optimizer = torch.optim.Adam(net.parameters(), lr = args.lr)
    for epoch in range(args.max_iter):
        t0 = time.time()

        net.train()
        emb = net(g, features)
        H = compute_H(emb, args.mu, device)
        loss = lossfunc(emb, H)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dur.append(time.time() - t0)
        print("Epoch {} | Loss {} | Time(s) {}".format(epoch, loss, np.mean(dur)))



    num_nodes = g.num_nodes()
    phi = emb.cpu().detach().numpy().T
    Emb = np.dot(phi, DCT(num_nodes))
    PATH = args.PATH
    torch.save(net, PATH) 
    filename = args.embeddingfile
    np.savetxt(filename, Emb)








