# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 23:18:04 2018

@author: pig84
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os
import time

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

def main():
    X_train, X_test, y_train, y_test = load_data('mnist_train.csv')
    n, d = X_train.shape
    n_test = X_test.shape[0]
    batch_size = 32
    epochs = 100
    learning_rate = 0.001
    inv_learning_rate = 0.001
    
    g_1 = tf.Graph()
    
    with g_1.as_default():
        with tf.device('/device:GPU:0'):
            with tf.name_scope('X_placeholder'):
                X_placeholder = tf.placeholder(tf.float32, [None, 784])
            with tf.name_scope('y_placeholder'):
                y_placeholder = tf.placeholder(tf.float32, [None, 10])
            
            X_resahpe = tf.reshape(X_placeholder, [-1, 28, 28, 1])
            
            #forward
            x1 = tf.layers.conv2d(
                    inputs = X_resahpe,
                    filters = 32,
                    kernel_size = 5,
                    padding = 'same',
                    activation = tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs = x1, pool_size=[2, 2], strides=2)
            pool1_flat = tf.reshape(pool1, [-1, 14*14*32])
            dense_1 = tf.layers.dense(pool1_flat, 256, activation = tf.nn.sigmoid)
            y1 = tf.layers.dense(y_placeholder, 256, activation = tf.nn.sigmoid)
            
            
            x2 = tf.layers.conv2d(
                    inputs = tf.stop_gradient(pool1),
                    filters = 64,
                    kernel_size = 5,
                    padding = 'same',
                    activation = tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs = x2, pool_size=[2, 2], strides=2)
            pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
            dense_2 = tf.layers.dense(pool2_flat, 256, activation = tf.nn.sigmoid)
            y2 = tf.layers.dense(tf.stop_gradient(y1), 256, activation = tf.nn.sigmoid)
            
            
            with tf.name_scope('y1_stop'):
                y1_stop = tf.stop_gradient(y1)
            with tf.name_scope('y2_stop'):
                y2_stop = tf.stop_gradient(y2)
            #with tf.name_scope('y3_stop'):
                #y3_stop = tf.stop_gradient(y3)
             
            with tf.name_scope('loss_1'):
                loss_1 = tf.losses.mean_squared_error(predictions = dense_1, labels = y1)
            with tf.name_scope('train_step_1'):
                train_step_1 = tf.train.AdamOptimizer(learning_rate).minimize(loss_1)
            
            with tf.name_scope('loss_2'):
                loss_2 = tf.losses.mean_squared_error(predictions = dense_2, labels = y2)
            with tf.name_scope('train_step_2'):
                train_step_2 = tf.train.AdamOptimizer(learning_rate).minimize(loss_2)
            
            
            #inverse 1   
            y_placeholder_hat = tf.layers.dense(y1, 10, activation = tf.nn.sigmoid)
            iw1 = tf.get_default_graph().get_tensor_by_name(os.path.split(y_placeholder_hat.name)[0] + '/kernel:0')
            ib1 = tf.get_default_graph().get_tensor_by_name(os.path.split(y_placeholder_hat.name)[0] + '/bias:0')
            with tf.name_scope('rev_loss_1'):
                rev_loss_1 = tf.losses.mean_squared_error(predictions = y_placeholder_hat, labels = y_placeholder)
            with tf.name_scope('rev_train_step_1'):
                rev_train_step_1 = tf.train.AdamOptimizer(inv_learning_rate).minimize(rev_loss_1)
            
            #inverse 2
            y1_hat = tf.layers.dense(y2, 256, activation = tf.nn.sigmoid)
            iw2 = tf.get_default_graph().get_tensor_by_name(os.path.split(y1_hat.name)[0] + '/kernel:0')
            ib2 = tf.get_default_graph().get_tensor_by_name(os.path.split(y1_hat.name)[0] + '/bias:0')
            with tf.name_scope('rev_loss_2'):
                rev_loss_2 = tf.losses.mean_squared_error(predictions = y1_hat, labels = y1_stop)
            with tf.name_scope('rev_train_step_2'):
                rev_train_step_2 = tf.train.AdamOptimizer(inv_learning_rate).minimize(rev_loss_2)
            
            train_step = [train_step_1, train_step_2]
            reverse_step = [rev_train_step_1, rev_train_step_2]
            
            #prediction
            with tf.name_scope('accuracy'):
                ia1 = tf.nn.sigmoid(tf.matmul(dense_2, iw2) + ib2)
                ia0 = tf.nn.sigmoid(tf.matmul(ia1, iw1) + ib1)
                pred = ia0
                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_placeholder, 1))
                correct_count = tf.cast(correct_prediction, tf.float32)
                accuracy = tf.reduce_mean(correct_count)
            
        #initializer
        init = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement = False, log_device_placement = True)
        config.gpu_options.allow_growth = True
        
        #saver
        saver = tf.train.Saver()
        with tf.Session(config = config) as sess:
            numbers = np.array([])
            sess.run(init)
            #saver.restore(sess, "./saver/model.ckpt")
            #print(sess.run(accuracy, feed_dict = {X_placeholder : X_test, y_placeholder : y_test}))
            #writer
            #writer = tf.summary.FileWriter('logs/', sess.graph)
            start_time = time.time()
            for epoch in range(epochs):
                score = 0
                for test_batch in range(int (n_test / batch_size)):
                    batch_xs_test = X_test[(test_batch*batch_size) : (test_batch+1)*batch_size]
                    batch_ys_test = y_test[(test_batch*batch_size) : (test_batch+1)*batch_size]
                    score += sum(sess.run(correct_count, feed_dict = {X_placeholder : batch_xs_test, y_placeholder : batch_ys_test}))
                for i in range(int (n_test / batch_size)*batch_size, n_test):
                    batch_xs_test = X_test[i].reshape(1, -1)
                    batch_ys_test = y_test[i].reshape(1, -1)
                    score += sum(sess.run(correct_count, feed_dict = {X_placeholder : batch_xs_test, y_placeholder : batch_ys_test}))
                print(epoch, score/n_test)
                numbers = np.append(numbers, score/n_test)
                
                for batch in range(int (n / batch_size)):
                    batch_xs = X_train[(batch*batch_size) : (batch+1)*batch_size]
                    batch_ys = y_train[(batch*batch_size) : (batch+1)*batch_size]
                    sess.run(train_step+reverse_step, feed_dict = {X_placeholder : batch_xs, y_placeholder : batch_ys})

            print(time.time()-start_time)

#            print(sess.run(accuracy, feed_dict = {X_placeholder : X_test, y_placeholder : y_test}))
            
            #saver.save(sess, "./saver/model.ckpt")
        np.savetxt('PCNN_accu.csv', numbers, delimiter=',')
        
if __name__ == '__main__':
    main()