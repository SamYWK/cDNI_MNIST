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
    np.seterr( over='ignore' ) #ignore sigmoid overflow
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
    
    def __init__(self,input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.random.randn(output_dim)
    
def network_training(q, count, Global_variable, X_train, y_train, process_id):
    n, d = X_train.shape
    batch_size = 200
    iterations = 50
    learning_rate = 0.001
    
    for iters in range(iterations):
        for batch in range(int(n / batch_size)):
            inputs = X_train[(batch*batch_size) : (batch+1)*batch_size]
            ground_truth = y_train[(batch*batch_size) : (batch+1)*batch_size]
            
            a1 = sigmoid(inputs.dot(Global_variable.W1) + Global_variable.b1) #input
            
            a2 = sigmoid(a1.dot(Global_variable.W2) + Global_variable.b2)

            z3 = a2.dot(Global_variable.W3) + Global_variable.b3
            a3 = stable_softmax(z3)
            
            ce = cross_entropy(logits = a3, labels = ground_truth) #loss
            
            if count.value % 10 == 0:
                loss = np.sum(ce)/200
                q.put(loss)
                print(process_id, ':', loss)
            count.value += 1

            layer_3_delta = a3 - ground_truth #gradient
            layer_2_delta = layer_3_delta.dot(Global_variable.W3.T)
            Global_variable.W3 -= a2.T.dot(layer_3_delta) * learning_rate
            Global_variable.b3 -= np.average(layer_3_delta, axis=0) * learning_rate
            
            layer_1_delta = layer_2_delta.dot(Global_variable.W2.T)
            Global_variable.W2 -= a1.T.dot(layer_2_delta) * learning_rate
            Global_variable.b2 -= np.average(layer_2_delta, axis=0) * learning_rate
            
            Global_variable.W1 -= inputs.T.dot(layer_1_delta) * learning_rate
            Global_variable.b1 -= np.average(layer_1_delta, axis=0) * learning_rate
            
    #print(process_id, 'process end.')

#inherit BaseManager
#class MyManager(BaseManager):
#    pass

#register shared class
#MyManager.register('network', network)

if __name__ == '__main__':
    print('Start')
    X_train, X_test, y_train, y_test = load_data('mnist_train.csv')
    n, d = X_train.shape
    process_num = 6
    section_size = int(n/process_num)
    np.random.seed(1)
    #start manager
    #manager = MyManager()
    #manager.start()
    
    #layer_1 = manager.network(d, 256, nonlin = sigmoid, nonlin_deriv = sigmoid_out2deriv)
    layer_1 = network(d, 256)
    layer_2 = network(256, 256)
    layer_3 = network(256, 10)
    
    #shared variables
    manager = mp.Manager()
    Global_variable = manager.Namespace()
    count = mp.Value('i', 0)
    
    Global_variable.W1 = layer_1.weights
    Global_variable.W2 = layer_2.weights
    Global_variable.W3 = layer_3.weights
    Global_variable.b1 = layer_1.bias
    Global_variable.b2 = layer_2.bias
    Global_variable.b3 = layer_3.bias
    
    #create queue
    mp_manager = mp.Manager()
    q = mp_manager.Queue()
    
    #create processes
    processes = []
    for process in range(process_num):
        p = mp.Process(target = network_training, args = (q, count, Global_variable,\
            X_train[(process*section_size) : (process+1)*section_size], y_train[(process*section_size) : (process+1)*section_size], process))
        processes.append(p)
    start_time = time.time()
    #start process
    for process in processes:
        process.start()
    
    #join process
    for process in processes:
        process.join()
    end_time = time.time()
    #save results
    results = []
    for _ in range(q.qsize()):
        results.append(q.get())
        
    np.savetxt('parallel_results.csv', np.asarray(results), delimiter=',')
    print("It cost %f sec" % (end_time-start_time))