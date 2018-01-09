# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 15:01:30 2018

@author: SamKao
"""

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time

def load_data(file_name):
    df = pd.read_csv(file_name)
    X = df.drop(['label'], axis = 1).values
    #normalize X
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    y = df['label'].values.reshape(-1, 1)
    enc = OneHotEncoder()
    y = enc.fit_transform(y).toarray()
    return train_test_split(X, y, test_size = 0.2, random_state = 1)

def cDNI(X_train, X_test, y_train, y_test):
    n, d = X_train.shape
    numbers = np.array([])
    batch_size = 100
    iterations = 100
    
    X_placeholder = tf.placeholder(tf.float32, [None, 784])
    y_placeholder = tf.placeholder(tf.float32, [None, 10])
    
    #add layers
    W1 = tf.Variable(tf.zeros([784, 100]))
    b1 = tf.Variable(tf.zeros([100]))
    z1 = tf.matmul(X_placeholder, W1)# + b1
    a1 = tf.nn.sigmoid(z1)
    
    W2 = tf.Variable(tf.random_normal([100, 100]))
    b2 = tf.Variable(tf.random_normal([100]))
    z2 = tf.matmul(a1, W2)# + b2
    a2 = tf.nn.sigmoid(z2)
    
    W3 = tf.Variable(tf.random_normal([100, 100]))
    b3 = tf.Variable(tf.random_normal([100]))
    z3 = tf.matmul(a2, W3)# + b3
    a3 = tf.nn.sigmoid(z3)
    
    W4 = tf.Variable(tf.random_normal([100, 100]))
    b4 = tf.Variable(tf.random_normal([100]))
    z4 = tf.matmul(a3, W4)# + b4
    a4 = tf.nn.sigmoid(z4)
    
    W5 = tf.Variable(tf.random_normal([100, 100]))
    b5 = tf.Variable(tf.random_normal([100]))
    z5 = tf.matmul(a4, W5)# + b5
    a5 = tf.nn.sigmoid(z5)
    
    W6 = tf.Variable(tf.random_normal([100, 100]))
    b6 = tf.Variable(tf.random_normal([100]))
    z6 = tf.matmul(a5, W6)# + b6
    a6 = tf.nn.sigmoid(z6)
    
    W7 = tf.Variable(tf.random_normal([100, 10]))
    b7 = tf.Variable(tf.random_normal([10]))
    z7 = tf.matmul(a6, W7)# + b7
    a7 = tf.nn.softmax(z7)
    
    #loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_placeholder, logits = z7))
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
    
    #prediction
    correct_prediction = tf.equal(tf.argmax(a7,1), tf.argmax(y_placeholder,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    init = tf.global_variables_initializer()
    
    with tf.Session(config = tf.ConfigProto(allow_soft_placement = True)) as sess:
        sess.run(init)
        for iters in range(iterations):
            for batch in range(int (n / batch_size)):
                batch_xs = X_train[(batch*batch_size) : (batch+1)*batch_size]
                batch_ys = y_train[(batch*batch_size) : (batch+1)*batch_size]
                sess.run(train_step, feed_dict = {X_placeholder: batch_xs, y_placeholder: batch_ys})
                if batch % 10 == 0:
                    print(sess.run(cross_entropy, feed_dict = {X_placeholder: batch_xs, y_placeholder: batch_ys}))
                    numbers = np.append(numbers, sess.run(cross_entropy, feed_dict={X_placeholder: batch_xs, y_placeholder: batch_ys}))
        print('Accuracy :', sess.run(accuracy, feed_dict={X_placeholder: X_test, y_placeholder: y_test}))
    np.savetxt('feed_forward_NN_numbers.csv', numbers, delimiter=',')
    
def main():
    print('')
    X_train, X_test, y_train, y_test = load_data('mnist_train.csv')
    start_time = time.time()
    cDNI(X_train, X_test, y_train, y_test)
    end_time = time.time()
    
    cost_time = end_time - start_time
    print ("It cost %f sec" % cost_time)
if __name__ == "__main__":
    main() 