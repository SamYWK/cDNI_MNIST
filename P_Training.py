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

def add_layer(inputs, in_dim, out_dim, activation = None, name = 'layer'):
    with tf.name_scope(name):
        with tf.name_scope('Weights'):
            W = tf.Variable(tf.truncated_normal([in_dim, out_dim]))
        with tf.name_scope('bias'):
            b = tf.Variable(tf.zeros([out_dim]))
        with tf.name_scope('Wx_plus_b'):
            if activation == None:
                output = tf.matmul(inputs, W) + b
            else:
                output = activation(tf.matmul(inputs, W) + b)
        return output, W, b

def main():
    X_train, X_test, y_train, y_test = load_data('mnist_train.csv')
    n, d = X_train.shape
    batch_size = 200
    epochs = 200
    learning_rate = 0.0025
    inv_epochs = 200
    inv_learning_rate = 0.0025
    g_1 = tf.Graph()
    
    with g_1.as_default():
        with tf.name_scope('inputs'):
            X_placeholder = tf.placeholder(tf.float32, [None, 784], name = 'x_inputs')
            y_placeholder = tf.placeholder(tf.float32, [None, 10], name = 'y_inputs')
        
        #forward
        x1, xW1, xb1 = add_layer(X_placeholder, 784, 256, activation = tf.nn.sigmoid, name = 'xlayer_1')
        y1, yW1, yb1 = add_layer(y_placeholder, 10, 256, activation = None, name = 'ylayer_1')
        
        x2, xW2, xb2 = add_layer(tf.stop_gradient(x1), 256, 256, activation = tf.nn.sigmoid, name = 'xlayer_2')
        y2, yW2, yb2 = add_layer(tf.stop_gradient(y1), 256, 256, activation = None, name = 'ylayer_2')
        
        x3, xW3, xb3 = add_layer(tf.stop_gradient(x2), 256, 256, activation = tf.nn.sigmoid, name = 'xlayer_3')
        y3, yW3, yb3 = add_layer(tf.stop_gradient(y2), 256, 256, activation = None, name = 'ylayer_3')
        
        with tf.name_scope('loss_1'):
            loss_1 = tf.reduce_mean(tf.reduce_sum(tf.square(x1 - y1), axis = 1))
        with tf.name_scope('train_step_1'):
            train_step_1 = tf.train.AdamOptimizer(learning_rate).minimize(loss_1)
        
        with tf.name_scope('loss_2'):
            loss_2 = tf.reduce_mean(tf.reduce_sum(tf.square(x2 - y2), axis = 1))
        with tf.name_scope('train_step_2'):
            train_step_2 = tf.train.AdamOptimizer(learning_rate).minimize(loss_2)
        
        with tf.name_scope('loss_3'):
            loss_3 = tf.reduce_mean(tf.reduce_sum(tf.square(x3 - y3), axis = 1))
        with tf.name_scope('train_step_3'):
            train_step_3 = tf.train.AdamOptimizer(learning_rate).minimize(loss_3)
        
        #inverse mapping
        #inverse 1
        y_placeholder_hat, iW1, ib1 = add_layer(tf.stop_gradient(y1), 256, 10, activation = None, name = 'invlayer_1')
        with tf.name_scope('rev_loss_1'):
            rev_loss_1 = tf.reduce_mean(tf.reduce_sum(tf.square(y_placeholder_hat - y_placeholder), axis = 1))
        with tf.name_scope('rev_train_step_1'):
            rev_train_step_1 = tf.train.AdamOptimizer(inv_learning_rate).minimize(rev_loss_1)
        
        #inverse 2
        y1_hat, iW2, ib2 = add_layer(tf.stop_gradient(y2), 256, 256, activation = None, name = 'invlayer_2')
        with tf.name_scope('y1_stop'):
            y1_stop = tf.stop_gradient(y1)
        with tf.name_scope('rev_loss_2'):
            rev_loss_2 = tf.reduce_mean(tf.reduce_sum(tf.square(y1_hat - y1_stop), axis = 1))
        with tf.name_scope('rev_train_step_2'):
            rev_train_step_2 = tf.train.AdamOptimizer(inv_learning_rate).minimize(rev_loss_2)
            
        #inverse 3
        y2_hat, iW3, ib3 = add_layer(tf.stop_gradient(y3), 256, 256, activation = None, name = 'invlayer_3')
        with tf.name_scope('y2_stop'):
            y2_stop = tf.stop_gradient(y2)
        with tf.name_scope('rev_loss_3'):
            rev_loss_3 = tf.reduce_mean(tf.reduce_sum(tf.square(y2_hat - y2_stop), axis = 1))
        with tf.name_scope('rev_train_step_3'):
            rev_train_step_3 = tf.train.AdamOptimizer(inv_learning_rate).minimize(rev_loss_3)
        
        train_step = [train_step_1, train_step_2, train_step_3]
        reverse_step = [rev_train_step_1, rev_train_step_2, rev_train_step_3]
        loss = [loss_1, loss_2, loss_3]
        reverse_loss = [rev_loss_1, rev_loss_2, rev_loss_3]
        
        #prediction
        with tf.name_scope('accuracy'):
            pred = tf.matmul((tf.matmul((tf.matmul(x3, iW3) + ib3), iW2) + ib2), iW1) + ib1
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_placeholder, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
        #initializer
        init = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = True)
        config.gpu_options.allow_growth = True
        
        with tf.Session(config = config) as sess:
            sess.run(init)
            print(sess.run(accuracy, feed_dict = {X_placeholder : X_test, y_placeholder : y_test}))
            #writer
            #writer = tf.summary.FileWriter('logs/', sess.graph)
            for epoch in range(epochs):
                for batch in range(int (n / batch_size)):
                    batch_xs = X_train[(batch*batch_size) : (batch+1)*batch_size]
                    batch_ys = y_train[(batch*batch_size) : (batch+1)*batch_size]
                    sess.run(train_step, feed_dict = {X_placeholder : batch_xs, y_placeholder : batch_ys})
                    if batch % 1000 == 0:
                        print(sess.run(loss, feed_dict = {X_placeholder : batch_xs, y_placeholder : batch_ys}))

            #train reverse auto-encoder
            for epoch in range(inv_epochs):
                for batch in range(int (n / batch_size)):
                    batch_xs = X_train[(batch*batch_size) : (batch+1)*batch_size]
                    batch_ys = y_train[(batch*batch_size) : (batch+1)*batch_size]
                    sess.run(reverse_step, feed_dict = {X_placeholder : batch_xs, y_placeholder : batch_ys})
                    if batch % 1000 == 0:
                        print(sess.run(reverse_loss, feed_dict = {X_placeholder : batch_xs, y_placeholder : batch_ys}))
                        #numbers = np.append(numbers, sess.run(loss, feed_dict={X_placeholder: batch_xs, y_placeholder: batch_ys}))
            print(sess.run(accuracy, feed_dict = {X_placeholder : X_test, y_placeholder : y_test}))
        #np.savetxt('cDNI_edit_numbers.csv', numbers, delimiter=',')
        
if __name__ == '__main__':
    main()