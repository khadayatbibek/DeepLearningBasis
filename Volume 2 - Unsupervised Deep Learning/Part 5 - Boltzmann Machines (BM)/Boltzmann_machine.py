# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 17:33:26 2020

@author: Bibek77
"""


#---------------Boltzmann_machine--------------------

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable 


#import the dataset

movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')


#preparing the traing set And test set 
training_set=pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set=np.array(training_set,dtype='int')

test_set=pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set=np.array(test_set,dtype='int')


# Getting the number of users and movies 
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

##converting the data into array with the user in line and movies in the column 

def convert(data):
    new_data=[]
    for id_users in range(1,nb_users + 1):
        id_movies=data[:,1][data[:,0]==id_users]#get all the movies id belonging to one userID
        id_ratings=data[:,2][data[:,0]==id_users]#get all the rating belonging to one userID
        ratings=np.zeros(nb_movies)
        ratings[id_movies -1]=id_ratings
        new_data.append(list(ratings))   
    return new_data 
training_set = convert(training_set)
test_set = convert(test_set)



#converting the data into torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)
 

#Converting  the rating into binary rating 1(like) or 0(unlike)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1


#Create the architecture of  Neural Network

class RBM():
    def __init__(self,no_visble_node,no_hidden_node):
        self.W=torch.randn(no_hidden_node,no_visble_node)
        self.a=torch.randn(1,no_hidden_node)
        self.b=torch.randn(1,no_visble_node)  
    def sample_hidden(self,x):
        wx=torch.mm(x,self.W.t())
        activation=wx + self.a.expand_as(wx)
        p_hidden_given_visible=torch.sigmoid(activation)
        return p_hidden_given_visible, torch.bernoulli(p_hidden_given_visible)
    def sample_visible(self,y):
       
        wy = torch.mm(y, self.W)
        
        activation = wy + self.b.expand_as(wy)
        p_visible_given_hidden=torch.sigmoid(activation)
        return p_visible_given_hidden, torch.bernoulli(p_visible_given_hidden) 
    def train(self, v0, vk, ph0, phk):       
        
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

no_visble_node = len(training_set[0])
no_hidden_node = 100
batch_size = 100
rbm = RBM(no_visble_node, no_hidden_node)




#Training RBM  model
nb_epoch=10

for epoch in range(1,nb_epoch+1):
    train_loss=0
    s=0.
    for id_user in range(0,nb_users-batch_size,batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_hidden(v0)
        for k in range(10):
            _,hk = rbm.sample_hidden(vk)#inverse trick 
            _,vk = rbm.sample_visible(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_hidden(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
    
        
        
# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_hidden(v)
        _,v = rbm.sample_visible(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))












