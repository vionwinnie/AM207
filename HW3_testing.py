# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 07:57:40 2017

@author: Dell
"""


import pandas as pd
import numpy as np

#import theano
#from theano import *
#import theano.tensor as T

df= pd.read_csv(r'C:\Users\Dell\Documents\Python Scripts\HW3_data.csv', sep=',',header=None)
df=df.as_matrix(columns=None)



#x = T.dmatrix('x')
#s = 1 / (1 + T.exp(-x))
#logistic = function([x], s)
#logistic([[0, 1], [-1, -2]])

def loss(lambda1,lambda2):
    x = df[0,:]
    y = df[1,:]
    loss_fun = np.zeros(len(x))
    def function(xi,yi):
        func = 0.000045*(lambda2**2)*yi - 0.000098*(lambda1**2)*xi + 0.003926*lambda1*np.exp((yi**2-xi**2)*(lambda1**2+lambda2**2))
        return func
    for i in range(len(loss_fun)):
        loss_x_y = function(x[i],y[i])
        loss_fun[i] = loss_x_y
    
    net_loss = np.sum(loss_fun)
    return net_loss

lambda1s = np.linspace(0,5,1000)
lambda2s = np.linspace(0,5,1000)
lam1, lam2 = np.meshgrid(lambda1s,lambda2s)
test = zip(np.ravel(lam1),np.ravel(lam2))
testx = [x for x in test]

costs = np.zeros(1000000)

for i in range(len(costs)):
    cost = loss(testx[i][0],testx[i][1])
    costs[i] = cost

print('completed')

