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
    batch_size = 200
    epochs = 200
    learning_rate = 0.001
    inv_epochs = 100
    inv_learning_rate = 0.005
    
    g_1 = tf.Graph()
    
    with g_1.as_default():
        with tf.device('/device:GPU:0'):
            with tf.name_scope('X_placeholder'):
                X_placeholder = tf.placeholder(tf.float32, [None, 784], name = 'x_inputs')
            with tf.name_scope('y_placeholder'):
                y_placeholder = tf.placeholder(tf.float32, [None, 10], name = 'y_inputs')
            
            #forward
            x1 = tf.layers.dense(X_placeholder, 256, activation = tf.nn.sigmoid, name = 'xlayer_1')
            w1 = tf.get_default_graph().get_tensor_by_name(os.path.split(x1.name)[0] + '/kernel:0')
            y1 = tf.layers.dense(y_placeholder, 256, activation = tf.nn.sigmoid, name = 'ylayer_1')
            
            x2 = tf.layers.dense(tf.stop_gradient(x1), 256, activation = tf.nn.sigmoid, name = 'xlayer_2')
            y2 = tf.layers.dense(tf.stop_gradient(y1), 256, activation = tf.nn.sigmoid, name = 'ylayer_2')
            
            x3 = tf.layers.dense(tf.stop_gradient(x2), 256, activation = tf.nn.sigmoid, name = 'xlayer_3')
            y3 = tf.layers.dense(tf.stop_gradient(y2), 256, activation = tf.nn.sigmoid, name = 'ylayer_3')
            
            
            with tf.name_scope('y1_stop'):
                y1_stop = tf.stop_gradient(y1)
            with tf.name_scope('y2_stop'):
                y2_stop = tf.stop_gradient(y2)
            with tf.name_scope('y3_stop'):
                y3_stop = tf.stop_gradient(y3)
                
            with tf.name_scope('loss_1'):
                loss_1 = tf.losses.mean_squared_error(predictions = x1, labels = y1)
            with tf.name_scope('train_step_1'):
                train_step_1 = tf.train.AdamOptimizer(learning_rate).minimize(loss_1)
            
            with tf.name_scope('loss_2'):
                loss_2 = tf.losses.mean_squared_error(predictions = x2, labels = y2)
            with tf.name_scope('train_step_2'):
                train_step_2 = tf.train.AdamOptimizer(learning_rate).minimize(loss_2)
            
            with tf.name_scope('loss_3'):
                loss_3 = tf.losses.mean_squared_error(predictions = x3, labels = y3)
            with tf.name_scope('train_step_3'):
                train_step_3 = tf.train.AdamOptimizer(learning_rate).minimize(loss_3)
            
        #with tf.device('/cpu:0'):
            #inverse mapping
            
            #inverse 1   
            y_placeholder_hat = tf.layers.dense(y1, 10, activation = tf.nn.sigmoid, name = 'invlayer_1')
            iw1 = tf.get_default_graph().get_tensor_by_name(os.path.split(y_placeholder_hat.name)[0] + '/kernel:0')
            ib1 = tf.get_default_graph().get_tensor_by_name(os.path.split(y_placeholder_hat.name)[0] + '/bias:0')
            with tf.name_scope('rev_loss_1'):
                rev_loss_1 = tf.losses.mean_squared_error(predictions = y_placeholder_hat, labels = y_placeholder)
            with tf.name_scope('rev_train_step_1'):
                rev_train_step_1 = tf.train.AdamOptimizer(inv_learning_rate).minimize(rev_loss_1)
            
            #inverse 2
            y1_hat = tf.layers.dense(y2, 256, activation = tf.nn.sigmoid, name = 'invlayer_2')
            iw2 = tf.get_default_graph().get_tensor_by_name(os.path.split(y1_hat.name)[0] + '/kernel:0')
            ib2 = tf.get_default_graph().get_tensor_by_name(os.path.split(y1_hat.name)[0] + '/bias:0')
            with tf.name_scope('rev_loss_2'):
                rev_loss_2 = tf.losses.mean_squared_error(predictions = y1_hat, labels = y1_stop)
            with tf.name_scope('rev_train_step_2'):
                rev_train_step_2 = tf.train.AdamOptimizer(inv_learning_rate).minimize(rev_loss_2)
                
            #inverse 3
            y2_hat = tf.layers.dense(y3, 256, activation = tf.nn.sigmoid, name = 'invlayer_3')
            iw3 = tf.get_default_graph().get_tensor_by_name(os.path.split(y2_hat.name)[0] + '/kernel:0')
            ib3 = tf.get_default_graph().get_tensor_by_name(os.path.split(y2_hat.name)[0] + '/bias:0')
            with tf.name_scope('rev_loss_3'):
                rev_loss_3 =  tf.losses.mean_squared_error(predictions = y2_hat, labels = y2_stop)
            with tf.name_scope('rev_train_step_3'):
                rev_train_step_3 = tf.train.AdamOptimizer(inv_learning_rate).minimize(rev_loss_3)
    
            train_step = [train_step_1, train_step_2, train_step_3]
            reverse_step = [rev_train_step_1, rev_train_step_2, rev_train_step_3]
            loss = [loss_1, loss_2, loss_3]
            reverse_loss = [rev_loss_1, rev_loss_2, rev_loss_3]
            
            #prediction
            with tf.name_scope('accuracy'):
                ia2 = tf.nn.sigmoid(tf.matmul(x3, iw3) + ib3)
                ia1 = tf.nn.sigmoid(tf.matmul(ia2, iw2) + ib2)
                ia0 = tf.nn.sigmoid(tf.matmul(ia1, iw1) + ib1)
                pred = ia0
                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_placeholder, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        #initializer
        init = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement = False, log_device_placement = True)
        config.gpu_options.allow_growth = True
        
        #saver
        saver = tf.train.Saver()
        with tf.Session(config = config) as sess:
            sess.run(init)
            #saver.restore(sess, "./saver/model.ckpt")
            print(sess.run(accuracy, feed_dict = {X_placeholder : X_test, y_placeholder : y_test}))
            #writer
            #writer = tf.summary.FileWriter('logs/', sess.graph)
            start_time = time.time()
            for epoch in range(epochs):
                for batch in range(int (n / batch_size)):
                    batch_xs = X_train[(batch*batch_size) : (batch+1)*batch_size]
                    batch_ys = y_train[(batch*batch_size) : (batch+1)*batch_size]
                    sess.run(train_step+reverse_step, feed_dict = {X_placeholder : batch_xs, y_placeholder : batch_ys})
                    
                    if batch % 500 == 0:
                        print(sess.run(accuracy, feed_dict = {X_placeholder : batch_xs, y_placeholder : batch_ys}))
            print(sess.run(tf.reduce_sum(w1), feed_dict = {X_placeholder : X_test, y_placeholder : y_test}))
            print(time.time()-start_time)
            '''
            #train reverse auto-encoder
            for epoch in range(inv_epochs):
                for batch in range(int (n / batch_size)):
                    batch_xs = X_train[(batch*batch_size) : (batch+1)*batch_size]
                    batch_ys = y_train[(batch*batch_size) : (batch+1)*batch_size]
                    sess.run(reverse_step, feed_dict = {X_placeholder : batch_xs,y_placeholder : batch_ys})
                    if batch % 500 == 0:
                        print(sess.run(reverse_loss, feed_dict = {X_placeholder : batch_xs, y_placeholder : batch_ys}))
                        #numbers = np.append(numbers, sess.run(loss, feed_dict={X_placeholder: batch_xs, y_placeholder: batch_ys}))
            '''
            #print(sess.run(accuracy, feed_dict = {X_placeholder : X_test, y_placeholder : y_test}))
            
            #saver.save(sess, "./saver/model.ckpt")
        #np.savetxt('cDNI_edit_numbers.csv', numbers, delimiter=',')
        
if __name__ == '__main__':
    main()