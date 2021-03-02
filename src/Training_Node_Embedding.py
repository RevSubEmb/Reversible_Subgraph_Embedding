#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
        
    def forward(self, input_tensor):
        n, p = input_tensor.size()
        Psi = torch.tensor(DCT(n),dtype=torch.float32).cuda()
        Phi = torch.mm(Psi,input_tensor)
        result = torch.mm(Phi, Phi.t())
        result[range(n), range(n)] -= 1
        return torch.norm(result, 'fro')
    
    def backward(self, grad_output):
        return grad_output


def load_cora_data():
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    g = dgl.from_networkx(data.graph)
    return g, features, labels

def load_citeseer_data():
    data = citegrh.load_citeseer()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    g = dgl.from_networkx(data.graph)
    return g, features, labels

def load_graph_data(data):
    g = data[0]
    num_class = data.num_classes
    features = g.ndata['feat']  
    labels = g.ndata['label']  
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


# In[2]:


# """
device = ''
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

batch_graphs, batch_features, graph_size = load_batch_graph('PTC')
# data = dgl.data.CiteseerGraphDataset()
# g, features, labels = load_graph_data(data)
# G = nx.read_adjlist('./dataset/Wiki-Vote.txt')
# g = dgl.from_networkx(G)
    
g = batch_graphs.to(device)
features = batch_features.to(device)
# features = torch.eye(sum(graph_size)).to(device)

g = g.to(device)
# features = torch.eye(g.num_nodes()).cuda()
features = features.to(device)
net = Net(features.shape[1],256).to(device)
lossfunc = NewLoss()
# """


dur = []

optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3)
for epoch in range(201):
    t0 = time.time()

    net.train()
    emb = net(g, features)
#     H = compute_H(emb, 0.01, device)
    loss = lossfunc(emb)
    

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    dur.append(time.time() - t0)
    if epoch%5 == 0:
        print("Epoch {} | Loss {} | Time(s) {}".format(epoch, loss, np.mean(dur)))


# In[14]:


# Test R.I.P
import time
# """
num_nodes = g.num_nodes()
phi = emb.cpu().detach().numpy().T
Emb = np.dot(phi, DCT(num_nodes))
# """
x = np.zeros(num_nodes)
index = np.random.choice(range(num_nodes),size=20)
x[index] = np.random.rand(20)+np.log(2)
y=np.dot(Emb,x)

def decode(y, Emb):
    lasso = Lasso(alpha = 1e-6, max_iter = 10000)
    lasso.fit(Emb, y)
    sol = np.zeros(lasso.coef_.size)
#     index1 = np.where(lasso.coef_>0.5)[0]
    index2 = np.where(lasso.coef_>0.25)[0]
    index1 = np.zeros(index2.size)
    while set(index1) != set(index2):
        index1 = index2
        C = np.linalg.lstsq(Emb[:,index1], y, rcond=None)
        sol = np.zeros(lasso.coef_.size)
        sol[index1]=C[0]
        index2 = np.where(sol>0.25)[0]
    return sol

tStart = time.time()
solution = decode(y,Emb)
tEnd = time.time()
print(x[x.nonzero()])
print(solution[solution.nonzero()])
print("Time used:",tEnd-tStart)


# In[18]:


acc = list()
err = list()
for i in range(100):
    test = np.zeros(num_nodes)
    index = np.random.choice(range(num_nodes),size=15)
    test[index] = np.random.rand(15)+np.log(2)
    y=np.dot(Emb,test)
    tmp = decode(y, Emb)
    error = np.linalg.norm(tmp-test)
    err.append(error)
    if error<1e-8:
        acc.append(1)
    else:
        acc.append(0)

print(sum(acc))


# In[19]:


# PATH = "./WikiVote.pth"
# torch.save(net, PATH) 
filename = "./Citeseer_NodeEmbedding_128.txt"
np.savetxt(filename,Emb)


# In[17]:


lasso = Lasso(alpha = 1e-7, max_iter = 10000)
lasso.fit(Emb, y)
print(x.nonzero()[0])
print(np.where(lasso.coef_>0.25)[0])


# In[167]:


g.num_nodes()


# In[ ]:




