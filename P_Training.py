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
    learning_rate = 0.00001
    np.random.seed(1)
    
    X_placeholder = tf.placeholder(tf.float32, [None, 784])
    y_placeholder = tf.placeholder(tf.float32, [None, 10])
    
    #layer 1
    x1 = tf.layers.dense(X_placeholder, 256, tf.nn.relu)
    y_transform_weight = tf.random_normal([10, 784])
    y_ = tf.stop_gradient(tf.matmul(y_placeholder, y_transform_weight))
    y1 = tf.layers.dense(y_, 256, trainable = None)
    loss_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y1, logits = x1))
    train_step_1 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_1)
    
    #layer 2
    stop_x1 = tf.stop_gradient(x1)
    stop_y1 = tf.stop_gradient(y1)
    x2, W2, b2 = add_layer(stop_x1, 256, 10, None)
    y2 = tf.nn.sigmoid(tf.matmul(stop_y1, W2) + b2)
    
    loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y2, logits = x2))
    train_step_2 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_2)
    
    #test error
    correct_prediction = tf.equal(tf.argmax(x2,1), tf.argmax(y_placeholder,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    #initializer
    init = tf.global_variables_initializer()
    
    with tf.Session(config = tf.ConfigProto(allow_soft_placement = True)) as sess:
        sess.run(init)
        for iters in range(iterations):
            for batch in range(int (n / batch_size)):
                batch_xs = X_train[(batch*batch_size) : (batch+1)*batch_size]
                batch_ys = y_train[(batch*batch_size) : (batch+1)*batch_size]
                
                sess.run([train_step_1], feed_dict = {X_placeholder : batch_xs, y_placeholder : batch_ys})
                if batch % 50 == 0:
                    print(sess.run([y_transform_weight], feed_dict = {X_placeholder : batch_xs, y_placeholder : batch_ys}))
                #print(sess.run(accuracy, feed_dict = {X_placeholder : batch_xs, y_placeholder : batch_ys}))

if __name__ == '__main__':
    main()