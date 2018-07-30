# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 14:25:00 2018

@author: SamKao
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os
import time
from multiprocessing import Process
from time import sleep

cluster = tf.train.ClusterSpec({"worker": [ "localhost:3333", "localhost:3334" ], "ps": ["localhost:3335"]})
batch_size = 32
epochs = 100
learning_rate = 0.001
inv_learning_rate = 0.001

def load_data(file_name):
    df = pd.read_csv(file_name)
    X = df.drop(['label'], axis = 1).values.astype(np.float32)
    #normalize X
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    y = df['label'].values.reshape(-1, 1)
    enc = OneHotEncoder()
    y = enc.fit_transform(y).toarray()
    return train_test_split(X, y, test_size = 0.2, random_state = 1)

def ps():
    print('ps running')
    with tf.device('/job:ps/task:0'):
        with tf.name_scope('X_placeholder'):
            X_placeholder = tf.placeholder(tf.float32, [None, 784])
        with tf.name_scope('y_placeholder'):
            y_placeholder = tf.placeholder(tf.float32, [None, 10])
        
        X_resahpe = tf.reshape(X_placeholder, [-1, 28, 28, 1], name = 'X_resahpe')
        
        #forward
        
        x1 = tf.layers.conv2d(
                inputs = X_resahpe,
                filters = 32,
                kernel_size = 5,
                padding = 'same',
                activation = tf.nn.relu,
                name = 'x1')
        pool1 = tf.layers.max_pooling2d(inputs = x1, pool_size=[2, 2], strides=2, name = 'pool1')
        pool1_flat = tf.reshape(pool1, [-1, 14*14*32], name = 'pool1_flat')
        dense_1 = tf.layers.dense(pool1_flat, 256, activation = tf.nn.sigmoid, name = 'dense_1')
        y1 = tf.layers.dense(y_placeholder, 256, activation = tf.nn.sigmoid, name = 'y1')
        
        x2 = tf.layers.conv2d(
                inputs = tf.stop_gradient(pool1),
                filters = 64,
                kernel_size = 5,
                padding = 'same',
                activation = tf.nn.relu,
                name = 'x2')
        pool2 = tf.layers.max_pooling2d(inputs = x2, pool_size=[2, 2], strides=2, name = 'pool2')
        pool2_flat = tf.reshape(pool2, [-1, 7*7*64], name = 'pool2_flat')
        dense_2 = tf.layers.dense(pool2_flat, 256, activation = tf.nn.sigmoid, name = 'dense_2')
        y2 = tf.layers.dense(tf.stop_gradient(y1), 256, activation = tf.nn.sigmoid, name = 'y2')
        
        
        with tf.name_scope('y1_stop'):
            y1_stop = tf.stop_gradient(y1)
        with tf.name_scope('y2_stop'):
            y2_stop = tf.stop_gradient(y2)
        #with tf.name_scope('y3_stop'):
            #y3_stop = tf.stop_gradient(y3)
            
        #inverse 1   
        y_placeholder_hat = tf.layers.dense(y1, 10, activation = tf.nn.sigmoid, name = 'y_placeholder_hat')
        iw1 = tf.get_default_graph().get_tensor_by_name(os.path.split(y_placeholder_hat.name)[0] + '/kernel:0')
        ib1 = tf.get_default_graph().get_tensor_by_name(os.path.split(y_placeholder_hat.name)[0] + '/bias:0')
        
        
        #inverse 2
        y1_hat = tf.layers.dense(y2, 256, activation = tf.nn.sigmoid, name = 'y1_hat')
        iw2 = tf.get_default_graph().get_tensor_by_name(os.path.split(y1_hat.name)[0] + '/kernel:0')
        ib2 = tf.get_default_graph().get_tensor_by_name(os.path.split(y1_hat.name)[0] + '/bias:0')
        
    with tf.device('/job:worker/task:0'):
        with tf.name_scope('loss_1'):
            loss_1 = tf.losses.mean_squared_error(predictions = dense_1, labels = y1_stop)
        with tf.name_scope('train_step_1'):
            train_step_1 = tf.train.AdamOptimizer(learning_rate).minimize(loss_1)
            
        with tf.name_scope('loss_2'):
            loss_2 = tf.losses.mean_squared_error(predictions = dense_2, labels = y2_stop)
        with tf.name_scope('train_step_2'):
            train_step_2 = tf.train.AdamOptimizer(learning_rate).minimize(loss_2)
    
    with tf.device('/job:worker/task:1'):
        with tf.name_scope('rev_loss_1'):
            rev_loss_1 = tf.losses.mean_squared_error(predictions = y_placeholder_hat, labels = y_placeholder)
        with tf.name_scope('rev_train_step_1'):
            rev_train_step_1 = tf.train.AdamOptimizer(inv_learning_rate).minimize(rev_loss_1)
            
        with tf.name_scope('rev_loss_2'):
            rev_loss_2 = tf.losses.mean_squared_error(predictions = y1_hat, labels = y1_stop)
        with tf.name_scope('rev_train_step_2'):
            rev_train_step_2 = tf.train.AdamOptimizer(inv_learning_rate).minimize(rev_loss_2)
        
        
    server = tf.train.Server(cluster, job_name="ps", task_index=0) 
    
    sess = tf.Session(target=server.target)
    sess.run(tf.global_variables_initializer())
    server.join()
    
def worker(worker_num, n, n_test, X_train, X_test, y_train, y_test):
    with tf.device('/job:ps/task:0'):
        with tf.name_scope('X_placeholder'):
            X_placeholder = tf.placeholder(tf.float32, [None, 784])
        with tf.name_scope('y_placeholder'):
            y_placeholder = tf.placeholder(tf.float32, [None, 10])
        
        X_resahpe = tf.reshape(X_placeholder, [-1, 28, 28, 1], name = 'X_resahpe')
        
        #forward
        
        x1 = tf.layers.conv2d(
                inputs = X_resahpe,
                filters = 32,
                kernel_size = 5,
                padding = 'same',
                activation = tf.nn.relu,
                name = 'x1')
        pool1 = tf.layers.max_pooling2d(inputs = x1, pool_size=[2, 2], strides=2, name = 'pool1')
        pool1_flat = tf.reshape(pool1, [-1, 14*14*32], name = 'pool1_flat')
        dense_1 = tf.layers.dense(pool1_flat, 256, activation = tf.nn.sigmoid, name = 'dense_1')
        y1 = tf.layers.dense(y_placeholder, 256, activation = tf.nn.sigmoid, name = 'y1')
        
        x2 = tf.layers.conv2d(
                inputs = tf.stop_gradient(pool1),
                filters = 64,
                kernel_size = 5,
                padding = 'same',
                activation = tf.nn.relu,
                name = 'x2')
        pool2 = tf.layers.max_pooling2d(inputs = x2, pool_size=[2, 2], strides=2, name = 'pool2')
        pool2_flat = tf.reshape(pool2, [-1, 7*7*64], name = 'pool2_flat')
        dense_2 = tf.layers.dense(pool2_flat, 256, activation = tf.nn.sigmoid, name = 'dense_2')
        y2 = tf.layers.dense(tf.stop_gradient(y1), 256, activation = tf.nn.sigmoid, name = 'y2')
        
        
        with tf.name_scope('y1_stop'):
            y1_stop = tf.stop_gradient(y1)
        with tf.name_scope('y2_stop'):
            y2_stop = tf.stop_gradient(y2)
        #with tf.name_scope('y3_stop'):
            #y3_stop = tf.stop_gradient(y3)
         
        #inverse 1   
        y_placeholder_hat = tf.layers.dense(y1, 10, activation = tf.nn.sigmoid, name = 'y_placeholder_hat')
        iw1 = tf.get_default_graph().get_tensor_by_name(os.path.split(y_placeholder_hat.name)[0] + '/kernel:0')
        ib1 = tf.get_default_graph().get_tensor_by_name(os.path.split(y_placeholder_hat.name)[0] + '/bias:0')
        
        
        #inverse 2
        y1_hat = tf.layers.dense(y2, 256, activation = tf.nn.sigmoid, name = 'y1_hat')
        iw2 = tf.get_default_graph().get_tensor_by_name(os.path.split(y1_hat.name)[0] + '/kernel:0')
        ib2 = tf.get_default_graph().get_tensor_by_name(os.path.split(y1_hat.name)[0] + '/bias:0')
        
    
    with tf.device('/job:worker/task:0'):
        with tf.name_scope('loss_1'):
            loss_1 = tf.losses.mean_squared_error(predictions = dense_1, labels = y1_stop)
        with tf.name_scope('train_step_1'):
            train_step_1 = tf.train.AdamOptimizer(learning_rate).minimize(loss_1)
        
        with tf.name_scope('loss_2'):
            loss_2 = tf.losses.mean_squared_error(predictions = dense_2, labels = y2_stop)
        with tf.name_scope('train_step_2'):
            train_step_2 = tf.train.AdamOptimizer(learning_rate).minimize(loss_2)
            
#        #prediction
#        with tf.name_scope('accuracy'):
#            ia1 = tf.nn.sigmoid(tf.matmul(dense_2, iw2) + ib2)
#            ia0 = tf.nn.sigmoid(tf.matmul(ia1, iw1) + ib1)
#            pred = ia0
#            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_placeholder, 1))
#            correct_count = tf.cast(correct_prediction, tf.float32)
#            accuracy = tf.reduce_mean(correct_count)
    with tf.device('/job:worker/task:1'):
        with tf.name_scope('rev_loss_1'):
            rev_loss_1 = tf.losses.mean_squared_error(predictions = y_placeholder_hat, labels = y_placeholder)
        with tf.name_scope('rev_train_step_1'):
            rev_train_step_1 = tf.train.AdamOptimizer(inv_learning_rate).minimize(rev_loss_1)
            
        with tf.name_scope('rev_loss_2'):
            rev_loss_2 = tf.losses.mean_squared_error(predictions = y1_hat, labels = y1_stop)
        with tf.name_scope('rev_train_step_2'):
            rev_train_step_2 = tf.train.AdamOptimizer(inv_learning_rate).minimize(rev_loss_2)
        
    train_step = [train_step_1, train_step_2]
    reverse_step = [rev_train_step_1, rev_train_step_2]
    config = tf.ConfigProto(allow_soft_placement = False, log_device_placement = True)
    config.gpu_options.allow_growth = True
    server = tf.train.Server(cluster, job_name="worker", task_index=worker_num, config=config) 
    sess = tf.Session(target=server.target)
    
    for epoch in range(epochs):
        for batch in range(int (n / batch_size)):
            batch_xs = X_train[(batch*batch_size) : (batch+1)*batch_size]
            batch_ys = y_train[(batch*batch_size) : (batch+1)*batch_size]
            
            if worker_num == 0:
                sess.run(train_step, feed_dict = {X_placeholder : batch_xs, y_placeholder : batch_ys})
            elif worker_num == 1:
                sess.run(reverse_step, feed_dict = {X_placeholder : batch_xs, y_placeholder : batch_ys})
            
            if batch % 500 == 0:
                print('Epoch :', epoch)
                print('Loss', sess.run([loss_1, loss_2], feed_dict = {X_placeholder : batch_xs, y_placeholder : batch_ys}))
                print('Re_Loss', sess.run([rev_loss_1, rev_loss_2], feed_dict = {X_placeholder : batch_xs, y_placeholder : batch_ys}))
#                score = 0
#                for test_batch in range(int (n_test / batch_size)):
#                    batch_xs_test = X_test[(test_batch*batch_size) : (test_batch+1)*batch_size]
#                    batch_ys_test = y_test[(test_batch*batch_size) : (test_batch+1)*batch_size]
#                    score += sum(sess.run(correct_count, feed_dict = {X_placeholder : batch_xs_test, y_placeholder : batch_ys_test}))
#                for i in range(int (n_test / batch_size)*batch_size, n_test):
#                    batch_xs_test = X_test[i].reshape(1, -1)
#                    batch_ys_test = y_test[i].reshape(1, -1)
#                    score += sum(sess.run(correct_count, feed_dict = {X_placeholder : batch_xs_test, y_placeholder : batch_ys_test}))
#                print(score/n_test)

def main():
    X_train, X_test, y_train, y_test = load_data('mnist_train.csv')
    
    n, d = X_train.shape
    n_test = X_test.shape[0]
    
    
    
    ps_proc = Process(target=ps)
    w1_proc = Process(target=worker, args=(0, n, n_test, X_train, X_test, y_train, y_test))
    w2_proc = Process(target=worker, args=(1, n, n_test, X_train, X_test, y_train, y_test))
    ps_proc.start()
    sleep(10)
    w1_proc.start()
    w2_proc.start()
    ps_proc.join()
    w1_proc.join()
    w2_proc.join()
        
if __name__ == '__main__':
    main()