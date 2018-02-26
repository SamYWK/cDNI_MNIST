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

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs, Weights, biases

def main():
    X_train, X_test, y_train, y_test = load_data('mnist_train.csv')
    n, d = X_train.shape
    batch_size = 200
    iterations = 100
    learning_rate = 0.001
    np.random.seed(1)
    
    X_placeholder = tf.placeholder(tf.float32, [None, 784])
    y_placeholder = tf.placeholder(tf.float32, [None, 10])
    
    #layer 1
    x1 = tf.layers.dense(X_placeholder, 256, tf.nn.relu)
    y_transform_weight = tf.Variable(tf.random_normal([10, 784]))
    y_ = tf.stop_gradient(tf.matmul(y_placeholder, y_transform_weight))
    y1 = tf.layers.dense(y_, 256, trainable = None, activation = tf.nn.relu)
    loss_1 = tf.losses.mean_squared_error(labels = y1, predictions = x1)
    train_step_1 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_1)
    
    #layer 2
    stop_x1 = tf.stop_gradient(x1)
    stop_y1 = tf.stop_gradient(y1)
    x2 = tf.layers.dense(stop_x1, 256, tf.nn.relu)
    y2 = tf.layers.dense(stop_y1, 256, trainable = None, activation = tf.nn.relu)
    loss_2 = tf.losses.mean_squared_error(labels = y2, predictions = x2)
    train_step_2 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_2)
    
    #layer 3
    stop_x2 = tf.stop_gradient(x2)
    stop_y2 = tf.stop_gradient(y2)
    x3 = tf.layers.dense(stop_x2, 256, tf.nn.relu)
    y3 = tf.layers.dense(stop_y2, 256, trainable = None, activation = tf.nn.relu)
    loss_3 = tf.losses.mean_squared_error(labels = y3, predictions = x3)
    train_step_3 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_3)
    
    #layer 4
    stop_x3 = tf.stop_gradient(x3)
    stop_y3 = tf.stop_gradient(y3)
    x4 = tf.layers.dense(stop_x3, 10, tf.nn.relu)
    y4 = tf.layers.dense(stop_y3, 10, trainable = None, activation = tf.nn.relu)
    loss_4 = tf.losses.mean_squared_error(labels = y4, predictions = x4)
    train_step_4 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_4)
    
    #test error
    correct_prediction = tf.equal(tf.argmax(x4,1), tf.argmax(y_placeholder,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    #initializer
    init = tf.global_variables_initializer()
    
    with tf.Session(config = tf.ConfigProto(allow_soft_placement = True)) as sess:
        sess.run(init)
        for iters in range(iterations):
            for batch in range(int (n / batch_size)):
                batch_xs = X_train[(batch*batch_size) : (batch+1)*batch_size]
                batch_ys = y_train[(batch*batch_size) : (batch+1)*batch_size]
                
                sess.run([train_step_1, train_step_2, train_step_3, train_step_4], feed_dict = {X_placeholder : batch_xs, y_placeholder : batch_ys})
                if batch % 50 == 0:
                    print(sess.run([loss_4], feed_dict = {X_placeholder : batch_xs, y_placeholder : batch_ys}))
                #print(sess.run(accuracy, feed_dict = {X_placeholder : batch_xs, y_placeholder : batch_ys}))

if __name__ == '__main__':
    main()