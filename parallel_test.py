# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 12:25:46 2018

@author: pig84
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import time

def load_data(file_name):
    df = pd.read_csv(file_name)
    X = df.drop(['label'], axis = 1).values.astype(np.float64)
    #normalize X
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    y = df['label'].values.reshape(-1, 1)
    enc = OneHotEncoder()
    y = enc.fit_transform(y).toarray()
    return train_test_split(X, y, test_size = 0.2, random_state = 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_out2deriv(out):
    return out * (1 - out)

def stable_softmax(X):
    n, d = X.shape
    exps = np.exp(X - np.max(X))
    exps_sum = np.sum(exps, axis = 1)
    for _ in range(n):
        exps[_, :] = exps[_, :] / exps_sum[_]
    return exps

def cross_entropy(logits, labels, epsilon = 1e-12):
    logits = np.clip(logits, epsilon, 1. - epsilon)
    ce = -labels*np.log(logits + 1e-9)
    return ce

class network(object):
    
    def __init__(self,input_dim, output_dim, nonlin = None, nonlin_deriv = None, alpha = 0.001):
        
        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.random.randn(output_dim)
    
        self.nonlin = nonlin
        self.nonlin_deriv = nonlin_deriv
        self.alpha = alpha
    
    def forward_pass(self, inputs):
        self.inputs = inputs
        if self.nonlin == None :
            self.outputs = self.inputs.dot(self.weights) + self.bias
        else:
            self.outputs = self.nonlin(self.inputs.dot(self.weights) + self.bias)
        return self.outputs
    
    def backward_pass(self,true_gradient):
        if self.nonlin_deriv == None:
            grad = true_gradient
        else:
            grad = true_gradient * self.nonlin_deriv(self.outputs)
        
        self.weights -= self.inputs.T.dot(grad) * self.alpha
        self.bias -= np.average(grad,axis=0) * self.alpha
        return grad.dot(self.weights.T)
    
def network_training(q, locks, layer_1, layer_2, layer_3, X_train, y_train, process_id):
    n, d = X_train.shape
    batch_size = 200
    iterations = 10
    
    #print(process_id, 'process start.')
    for iters in range(iterations):
        for batch in range(int(n / batch_size)):
            locks[0].acquire()
            #print(process_id, 'stage_1.')
            a1 = layer_1.forward_pass(X_train[(batch*batch_size) : (batch+1)*batch_size]) #input
            locks[0].release()
               
            locks[1].acquire()
            #print(process_id, 'stage_2.') 
            a2 = layer_2.forward_pass(a1)
            locks[1].release()
                
            locks[2].acquire()
            #print(process_id, 'stage_3.')
            z3 = layer_3.forward_pass(a2)
            a3 = stable_softmax(z3)
            locks[2].release()
            
            locks[3].acquire()
            #print(process_id, 'stage_4.')  
            ce = cross_entropy(logits = a3, labels = y_train[(batch*batch_size) : (batch+1)*batch_size]) #loss
                        
            if batch % 10 == 0:
                loss = np.sum(ce)/200
                q.put(loss)
                print(process_id, ':', loss)
                        
            layer_3_delta = a3 - y_train[(batch*batch_size) : (batch+1)*batch_size] #gradient
            layer_2_delta = layer_3.backward_pass(layer_3_delta)
            locks[3].release()
        
            locks[4].acquire()
            #print(process_id, 'stage_5.')
            layer_1_delta = layer_2.backward_pass(layer_2_delta)
            locks[4].release()
                
            locks[5].acquire()
            #print(process_id, 'stage_6.')
            layer_1.backward_pass(layer_1_delta)
            locks[5].release()
            
    #print(process_id, 'process end.')

#inherit BaseManager
class MyManager(BaseManager):
    pass

#register shared class
MyManager.register('network', network)

if __name__ == '__main__':
    print('Start')
    X_train, X_test, y_train, y_test = load_data('mnist_train.csv')
    n, d = X_train.shape
    process_num = 6
    section_size = int(n/process_num)
    
    #start manager
    manager = MyManager()
    manager.start()
    
    layer_1 = manager.network(d, 256, nonlin = sigmoid, nonlin_deriv = sigmoid_out2deriv)
    layer_2 = manager.network(256, 256, nonlin = sigmoid, nonlin_deriv = sigmoid_out2deriv)
    layer_3 = manager.network(256, 10, nonlin = None, nonlin_deriv = None)
    
    #create locks
    locks = []
    for _ in range(6):
        locks.append(mp.Lock())
    #create queue
    mp_manager = mp.Manager()
    q = mp_manager.Queue()
        
    #create processes
    processes = []
    for process in range(process_num):
        p = mp.Process(target = network_training, args = (q, locks, layer_1, layer_2, layer_3,\
            X_train[(process*section_size) : (process+1)*section_size], y_train[(process*section_size) : (process+1)*section_size], process))
        processes.append(p)
    
    #start process
    for process in processes:
        process.start()
    
    #join process
    for process in processes:
        process.join()
    
    #save results
    results = []
    for _ in range(q.qsize()):
        results.append(q.get())
        
    np.savetxt('parallel_results.csv', np.asarray(results), delimiter=',')