# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 00:07:19 2017

@author: a560304
"""

#this code contains the part with early stopping 
import theano
from theano import *
import theano.tensor as T
import pandas as pd
import numpy as np

import gzip, pickle, time
import numpy as np
import matplotlib.pyplot as plt

#======================#
#loading the dataset into notebook as well as theano GPU#
#======================#
dataset=r"P:\Coursera\AM207\mnist.pkl.gz"

with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')
   
test_set_x, test_set_y = shared_dataset(test_set)
valid_set_x, valid_set_y = shared_dataset(valid_set)
train_set_x, train_set_y = shared_dataset(train_set)


#=====================#
#set up parameters #
#=====================#
batch_size = 256
n_train_batches = train_set[0].shape[0] // batch_size
n_valid_batches = valid_set[0].shape[0] // batch_size
n_test_batches = test_set[0].shape[0] // batch_size
learning_rate = 0.1

L2_set = [0.01,0.05,0.1,1]

avg_cost_early_stop = []
patience_num = []

for L2_reg in L2_set:
    
    print(L2_reg)
    #creating the logistic regression class #
    class LogisticRegression(object):
    
        def __init__(self, input, n_in, n_out):
            # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
            self.W = theano.shared(
                value=np.zeros(
                    (n_in, n_out),
                    dtype=theano.config.floatX
                ),
                name='W',
                borrow=True
            )
            # initialize the biases b as a vector of n_out 0s
            self.b = theano.shared(
                value=np.zeros(
                    (n_out,),
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
    
            # symbolic expression for computing the matrix of class-membership
            # probabilities
            self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
    
            # symbolic description of how to compute prediction as class whose
            # probability is maximal
            self.y_pred = T.argmax(self.p_y_given_x, axis=1)
            # end-snippet-1
    
            # parameters of the model
            self.params = [self.W, self.b]
    
            # keep track of model input
            self.input = input
            
            #computing sum of squares of paramters 
            self.L2_sqr = (self.W ** 2).sum()
             
        def negative_log_likelihood(self, y):
            """Return the mean of the negative log-likelihood of the prediction
            of this model under a given target distribution.
    
            """
    
            return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
            # end-snippet-2
    
        def errors(self, y):
            """Return a float representing the number of errors in the minibatch
            over the total number of examples of the minibatch ; zero one
            loss over the size of the minibatch
    
            """
    
            # check if y has same dimension of y_pred
            if y.ndim != self.y_pred.ndim:
                raise TypeError(
                    'y should have the same shape as self.y_pred',
                    ('y', y.type, 'y_pred', self.y_pred.type)
                )
            # check if y is of the correct datatype
            if y.dtype.startswith('int'):
                # the T.neq operator returns a vector of 0s and 1s, where 1
                # represents a mistake in prediction
                return T.mean(T.neq(self.y_pred, y))
            else:
                raise NotImplementedError()
    
                
        
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels
    
    #building on top of logistic regression 
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)
    
    
    #cost function with L2 Regularization of lambda = 0.01 
    cost = (classifier.negative_log_likelihood(y)
            + L2_reg * classifier.L2_sqr)
    
    #taking the cost function to evaluate gradient
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)
    #updating the weights and errors vector 
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]
    
    # creating a training function that computes the cost and updates the parameter of the model based on the rules
    index = T.lscalar()
    
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
    validate_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
    
    # recording the time at the beginning:
    start_time = time.time()
    
    #Run the model batch by batch and see how average cost decrease over time 
    avg_cost = []
    
    #initialize running the model 
    epoch = 0
    n_epochs=100
    validation_frequency = 100
    best_validation_loss = np.inf
    improvement_threshold = 0.995
    patience_increase = 2
    patience = 5000
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
    
            minibatch_avg_cost = train_model(minibatch_index)
            
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
    
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                avg_cost.append(this_validation_loss)
                
                if this_validation_loss < best_validation_loss:

                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    
                    best_validation_loss = this_validation_loss

                        
                print(
                    'patience %i,%i ,epoch %i, minibatch %i/%i, validation error %f %%' %
                    (   patience,   
                        L2_reg*100,
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    ))
        if patience <= iter:
            patience_num.append(iter)
            done_looping = True
            avg_cost_early_stop.append(avg_cost)
            break
        
plt.figure()
plt.semilogx(range(len(avg_cost_set[0])),avg_cost_set[0],'r',label=u'$\lambda = 0.005$')
plt.semilogx(range(len(avg_cost_set[1])),avg_cost_set[1],'b',label=u'$\lambda = 0.01$')
plt.semilogx(range(len(avg_cost_set[2])),avg_cost_set[2],'g',label=u'$\lambda = 0.05$')
plt.semilogx(range(len(avg_cost_set[3])),avg_cost_set[3],'m',label=u'$\lambda = 0.1$')
#plt.semilogx(range(len(avg_cost_early_stop[0])),avg_cost_early_stop[0],'r')
#plt.semilogx(range(len(avg_cost_early_stop[1])),avg_cost_early_stop[1],'b')
#plt.semilogx(range(len(avg_cost_early_stop[2])),avg_cost_early_stop[2],'g')
#plt.semilogx(range(len(avg_cost_early_stop[3])),avg_cost_early_stop[3],'m')

plt.xlim([0,200])
plt.xlabel('Iterations')
plt.ylabel('Validation Loss')
plt.axvline(x=50, linewidth=2, color='k',ls='dashed')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

