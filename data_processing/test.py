'''
# https://blog.csdn.net/qq_43601378/article/details/118759165
import torch
x = torch.tensor([[0.2,0.3,0.6,0.7]])
Pi = torch.nn.functional.softmax(x, dim=1)
print(Pi)

y = torch.tensor([3])
y = torch.nn.functional.one_hot(y)
print(y)

loss = y*torch.log(Pi+0.0000001)
print(loss)

loss = torch.sum(loss, dim=1)
print(loss)

loss = -torch.mean(loss, dim=0)
print(loss)


class Our_CrossEntropy(torch.nn.Module):

    def __init__(self):
        super(Our_CrossEntropy,self).__init__()
    
    def forward(self, x ,y):
        P_i = torch.nn.functional.softmax(x, dim=1)
        y = torch.nn.functional.one_hot(y)
        loss = y*torch.log(P_i + 0.0000001)
        loss = -torch.mean(torch.sum(loss, dim=1),dim = 0)
        return loss

our_loss = Our_CrossEntropy()
x1 = torch.tensor([[0.2,0.3,0.6,0.7],[0.3,0.4,0.8,0.2]])
y1 = torch.tensor([3,2])
our_loss_print = our_loss(x1,y1)

py_loss = torch.nn.CrossEntropyLoss()
py_loss_print = py_loss(x1,y1)

print("our loss:{}  pytorch loss:{}".format(our_loss_print, py_loss_print))

class one_hot_CrossEntropy(torch.nn.Module):

    def __init__(self):
        super(one_hot_CrossEntropy,self).__init__()
    
    def forward(self, x ,y):
        P_i = torch.nn.functional.softmax(x, dim=1)
        loss = y*torch.log(P_i + 0.0000001)
        loss = -torch.mean(torch.sum(loss,dim=1),dim = 0)
        return loss

x1 = torch.tensor([[0.2,0.3,0.6,0.7],[0.3,0.4,0.8,0.2]])
y1 = torch.tensor([[0,0,0,1],[0,0,1,0]])
onehot_loss = one_hot_CrossEntropy()
print(onehot_loss(x1,y1))
'''

# import numpy as np

# labels  = [1, 3, 4, 8, 7, 5, 2, 9, 0, 8, 7]
# one_hot_index = np.arange(len(labels)) * 10 + labels

# print ('one_hot_index:{}'.format(one_hot_index))

# one_hot = np.zeros((len(labels), 10))
# print(one_hot)
# one_hot.flat[one_hot_index] = 1

# print('one_hot:{}'.format(one_hot))


import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from training.train_utils import AverageMeter, ProgressMeter, accuracy
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.nn.functional as F


class AllReduce(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
    
# query:    [batch, 表征维度]
# supports: [原型数量，表征维度]
# return:   [batch, 原型数量]
def snn(query, supports, support_labels, temp=0.1):
    """ Soft Nearest Neighbours similarity classifier """
    softmax = torch.nn.Softmax(dim=1)
    query = torch.nn.functional.normalize(query)
    supports = torch.nn.functional.normalize(supports)
    # print("*************query", query.shape)
    # print("*************supports", supports.shape)
    aa = softmax(query @ supports / temp) @ support_labels
    # print("****************aa", aa.shape)
    return softmax(query @ supports / temp) @ support_labels
    
def one_hot(targets, num_classes, smoothing=0):
    device = torch.device('cuda:0')
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    targets = targets.long().view(-1, 1).to(device)
    return torch.full((len(targets), num_classes), off_value, device=device).scatter_(1, targets, on_value)

def sharpen(p, T):
    sharp_p = p**(1./T)
    sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
    return sharp_p

@torch.no_grad()
def distributed_sinkhorn(Q, num_itr=3, use_dist=True):
    _got_dist = use_dist and torch.distributed.is_available() \
        and torch.distributed.is_initialized() \
        and (torch.distributed.get_world_size() > 1)

    if _got_dist:
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1

    Q = Q.T
    B = Q.shape[1] * world_size  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    if _got_dist:
        torch.distributed.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(num_itr):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if _got_dist:
            torch.distributed.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.T

class x_hot_CrossEntropy(torch.nn.Module):
    def __init__(self):
        super(x_hot_CrossEntropy,self).__init__()
    
    def forward(self, x ,y):
        P_i = torch.nn.functional.softmax(x, dim=1)

        loss = y*torch.log(P_i + 0.0000001)
        loss = -torch.mean(torch.sum(loss,dim=1),dim = 0)
        return loss

proto_labels = one_hot(torch.tensor([i for i in range(256)]), 256)
probs = snn(q_pred, Memory_Bank.W, proto_labels)
with torch.no_grad():
    targets = sharpen(snn(k, Memory_Bank.W, proto_labels), T=0.25) 
    targets = distributed_sinkhorn(targets)
    targets = torch.cat([targets for _ in range(1)], dim=0)
bloss = torch.mean(torch.sum(torch.log(probs**(-targets)), dim=1))  # cross-entropy损失H(targets, queries)
avg_probs = AllReduce.apply(torch.mean(probs, dim=0))
rloss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))  
memax_loss = bloss + rloss