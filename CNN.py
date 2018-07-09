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
#import os
import time
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"

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
            
            x2 = tf.layers.conv2d(
                    inputs = pool1,
                    filters = 64,
                    kernel_size = 5,
                    padding = 'same',
                    activation = tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs = x2, pool_size=[2, 2], strides=2)
            pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
            
            x3 = tf.layers.dense(pool2_flat, (14*14*32), activation = tf.nn.sigmoid)
            x4 = tf.layers.dense(x3, 10, activation = tf.nn.sigmoid)
            
            with tf.name_scope('loss'):
                loss = tf.losses.mean_squared_error(predictions = x4, labels = y_placeholder)
            with tf.name_scope('train_step'):
                train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
                
            #prediction
            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(x4, 1), tf.argmax(y_placeholder, 1))
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
            
            start_time = time.time()
            for epoch in range(epochs):
                for batch in range(int (n / batch_size)):
                    batch_xs = X_train[(batch*batch_size) : (batch+1)*batch_size]
                    batch_ys = y_train[(batch*batch_size) : (batch+1)*batch_size]
                    sess.run(train_step, feed_dict = {X_placeholder : batch_xs, y_placeholder : batch_ys})
                    
                    if batch % 500 == 0:
                        score = 0
                        for test_batch in range(int (n_test / batch_size)):
                            batch_xs_test = X_test[(test_batch*batch_size) : (test_batch+1)*batch_size]
                            batch_ys_test = y_test[(test_batch*batch_size) : (test_batch+1)*batch_size]
                            score += sum(sess.run(correct_count, feed_dict = {X_placeholder : batch_xs_test, y_placeholder : batch_ys_test}))
                        for i in range(int (n_test / batch_size)*batch_size, n_test):
                            batch_xs_test = X_test[i].reshape(1, -1)
                            batch_ys_test = y_test[i].reshape(1, -1)
                            score += sum(sess.run(correct_count, feed_dict = {X_placeholder : batch_xs_test, y_placeholder : batch_ys_test}))
                        print(score/n_test)
                        numbers = np.append(numbers, score/n_test)
                        
            print(time.time()-start_time)
            
            #print(sess.run(accuracy, feed_dict = {X_placeholder : X_test, y_placeholder : y_test}))
            
            #saver.save(sess, "./saver/model.ckpt")
        np.savetxt('CNN.csv', numbers, delimiter=',')
        
if __name__ == '__main__':
    main()