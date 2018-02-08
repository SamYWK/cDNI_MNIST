# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:31:49 2018

@author: pig84
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
    X = df.drop(['label'], axis = 1).values.astype(np.float64)
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
    batch_size = 200
    learning_rate = 0.001
    iterations = 10
    
    X_placeholder = tf.placeholder(tf.float32, [None, 784])
    y_placeholder = tf.placeholder(tf.float32, [None, 10])
    
    #add layers
    W1 = tf.Variable(tf.truncated_normal([784, 256], stddev = 0.1))
    b1 = tf.Variable(tf.random_normal([256]))
    z1 = tf.matmul(X_placeholder, W1) + b1
    a1 = tf.nn.sigmoid(z1)
    
    W2 = tf.Variable(tf.truncated_normal([256, 256], stddev = 0.1))
    b2 = tf.Variable(tf.random_normal([256]))
    z2 = tf.matmul(a1, W2) + b2
    a2 = tf.nn.sigmoid(z2)
    
    W3 = tf.Variable(tf.truncated_normal([256, 10], stddev = 0.1))
    b3 = tf.Variable(tf.random_normal([10]))
    z3 = tf.matmul(a2, W3) + b3
    a3 = tf.nn.softmax(z3)
    
    #loss
    soft = tf.nn.softmax_cross_entropy_with_logits(labels = y_placeholder, logits = z3)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_placeholder, logits = z3))
    #train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    
    #True gradients
    W3_gradients = tf.gradients(ys = cross_entropy, xs = z3)[0]
    
    #synthetic gradients
    s_W1 = tf.Variable(tf.zeros([266, 256]))
    s_b1 = tf.Variable(tf.zeros([256]))
    s_z1 = tf.matmul(tf.concat([a1, y_placeholder], 1), s_W1) + s_b1
    
    s_W2 = tf.Variable(tf.zeros([266, 256]))
    s_b2 = tf.Variable(tf.zeros([256]))
    s_z2 = tf.matmul(tf.concat([a2, y_placeholder], 1), s_W2) + s_b2
    
    #Layer Weights Update
    W1_delta = tf.matmul(tf.transpose(X_placeholder), (a1 * (1 - a1) * s_z1))
    W1_update = W1.assign(W1 - learning_rate * W1_delta)
    W2_delta = tf.matmul(tf.transpose(a1), (a2 * (1 - a2) * s_z2))
    W2_update = W2.assign(W2 - learning_rate * W2_delta)
    W3_delta = tf.matmul(tf.transpose(a2), W3_gradients)#True gradients
    W3_update = W3.assign(W3 - learning_rate * W3_delta)
    
    #synthetic Weights Update
    
    l2_true_gradients = tf.matmul(W3_gradients, tf.transpose(W3))
    l2_s_error = tf.reduce_mean(tf.reduce_sum(tf.square(s_z2 - l2_true_gradients), axis = 1))
    s_W2_delta = tf.reshape(tf.gradients(ys = l2_s_error, xs = s_W2), [266, 256])
    s_W2_update = s_W2.assign(s_W2 - learning_rate * s_W2_delta)
    
    l1_true_gradients = tf.matmul(s_z2, tf.transpose(W2))
    l1_s_error = tf.reduce_mean(tf.reduce_sum(tf.square(s_z1 - l1_true_gradients), axis = 1))
    s_W1_delta = tf.reshape(tf.gradients(ys = l1_s_error, xs = s_W1), [266, 256])
    s_W1_update = s_W1.assign(s_W1 - learning_rate * s_W1_delta)
    
    #prediction
    correct_prediction = tf.equal(tf.argmax(a3,1), tf.argmax(y_placeholder,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    init = tf.global_variables_initializer()
    
    with tf.Session(config = tf.ConfigProto(allow_soft_placement = True)) as sess:
        sess.run(init)
        for iters in range(iterations):
            for batch in range(int (n / batch_size)):
                batch_xs = X_train[(batch*batch_size) : (batch+1)*batch_size]
                batch_ys = y_train[(batch*batch_size) : (batch+1)*batch_size]
                sess.run([W1_update, W2_update, W3_update,\
                          s_W2_update, s_W1_update], feed_dict = {X_placeholder: batch_xs, y_placeholder: batch_ys})
                #sess.run(train_step, feed_dict = {X_placeholder: batch_xs, y_placeholder: batch_ys})
                if batch % 50 == 0:
                    print(sess.run(cross_entropy, feed_dict = {X_placeholder: batch_xs, y_placeholder: batch_ys}))
                    numbers = np.append(numbers, sess.run(cross_entropy, feed_dict={X_placeholder: batch_xs, y_placeholder: batch_ys}))
        print('Accuracy :', sess.run(accuracy, feed_dict={X_placeholder: X_test, y_placeholder: y_test}))
    np.savetxt('cDNI_edit_numbers.csv', numbers, delimiter=',')
    
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