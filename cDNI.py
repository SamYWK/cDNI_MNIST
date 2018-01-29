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

def activation_diff(z, a):
    return tf.gradients(ys = a, xs = z)[0]

def cDNI(X_train, X_test, y_train, y_test):
    n, d = X_train.shape
    numbers = np.array([])
<<<<<<< HEAD
    batch_size = 256
    learning_rate = 0.00003
=======
    batch_size = 200
    learning_rate = 0.001
>>>>>>> c92552247a33267364b3a8a5e26a74508c7859f8
    iterations = 100
    epsilon = 0.0001
    
    
    X_placeholder = tf.placeholder(tf.float32, [None, 784])
    y_placeholder = tf.placeholder(tf.float32, [None, 10])
    
    #add layers
<<<<<<< HEAD
    W1 = tf.Variable(tf.truncated_normal([784, 256], stddev = 0.1))
    b1 = tf.Variable(tf.zeros([256]))
=======
    W1 = tf.Variable(tf.random_normal([784, 256]))
    b1 = tf.Variable(tf.random_normal([256]))
>>>>>>> c92552247a33267364b3a8a5e26a74508c7859f8
    z1 = tf.matmul(X_placeholder, W1) + b1
    mean_1, var_1 = tf.nn.moments(z1, axes = [0])
    scale_1 = tf.Variable(tf.ones([256]))
    shift_1 = tf.Variable(tf.zeros([256]))
    n1 = tf.nn.batch_normalization(z1, mean_1, var_1, shift_1, scale_1, epsilon)
    a1 = tf.nn.relu(n1)
    
    W2 = tf.Variable(tf.truncated_normal([256, 256], stddev = 0.1))
    b2 = tf.Variable(tf.random_normal([256]))
    z2 = tf.matmul(a1, W2) + b2
    mean_2, var_2 = tf.nn.moments(z2, axes = [0])
    scale_2 = tf.Variable(tf.ones([256]))
    shift_2 = tf.Variable(tf.zeros([256]))
    n2 = tf.nn.batch_normalization(z2, mean_2, var_2, shift_2, scale_2, epsilon)
    a2 = tf.nn.relu(n2)
    
    W3 = tf.Variable(tf.truncated_normal([256, 256], stddev = 0.1))
    b3 = tf.Variable(tf.random_normal([256]))
    z3 = tf.matmul(a2, W3) + b3
    mean_3, var_3 = tf.nn.moments(z3, axes = [0])
    scale_3 = tf.Variable(tf.ones([256]))
    shift_3 = tf.Variable(tf.zeros([256]))
    n3 = tf.nn.batch_normalization(z3, mean_3, var_3, shift_3, scale_3, epsilon)
    a3 = tf.nn.relu(n3)
    
    W4 = tf.Variable(tf.truncated_normal([256, 256], stddev = 0.1))
    b4 = tf.Variable(tf.random_normal([256]))
    z4 = tf.matmul(a3, W4) + b4
    mean_4, var_4 = tf.nn.moments(z4, axes = [0])
    scale_4 = tf.Variable(tf.ones([256]))
    shift_4 = tf.Variable(tf.zeros([256]))
    n4 = tf.nn.batch_normalization(z4, mean_4, var_4, shift_4, scale_4, epsilon)
    a4 = tf.nn.relu(n4)
    
    W5 = tf.Variable(tf.truncated_normal([256, 256], stddev = 0.1))
    b5 = tf.Variable(tf.random_normal([256]))
    z5 = tf.matmul(a4, W5) + b5
    mean_5, var_5 = tf.nn.moments(z5, axes = [0])
    scale_5 = tf.Variable(tf.ones([256]))
    shift_5 = tf.Variable(tf.zeros([256]))
    n5 = tf.nn.batch_normalization(z5, mean_5, var_5, shift_5, scale_5, epsilon)
    a5 = tf.nn.relu(n5)
    
    W6 = tf.Variable(tf.truncated_normal([256, 10], stddev = 0.1))
    b6 = tf.Variable(tf.random_normal([10]))
    z6 = tf.matmul(a5, W6) + b6
    a6 = tf.nn.softmax(z6)
    
    #loss
    soft = tf.nn.softmax_cross_entropy_with_logits(labels = y_placeholder, logits = z6)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_placeholder, logits = z6))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    
    #True gradients
    W6_gradients = tf.gradients(ys = cross_entropy, xs = z6)[0]
    
    #synthetic gradients
    s_W1 = tf.Variable(tf.zeros([266, 256]))
    s_b1 = tf.Variable(tf.zeros([256]))
    s_z1 = tf.matmul(tf.concat([a1, y_placeholder], 1), s_W1) + s_b1
    
    s_W2 = tf.Variable(tf.zeros([266, 256]))
    s_b2 = tf.Variable(tf.zeros([256]))
    s_z2 = tf.matmul(tf.concat([a2, y_placeholder], 1), s_W2) + s_b2
    
    s_W3 = tf.Variable(tf.zeros([266, 256]))
    s_b3 = tf.Variable(tf.zeros([256]))
    s_z3 = tf.matmul(tf.concat([a3, y_placeholder], 1), s_W3) + s_b3
    
    s_W4 = tf.Variable(tf.zeros([266, 256]))
    s_b4 = tf.Variable(tf.zeros([256]))
    s_z4 = tf.matmul(tf.concat([a4, y_placeholder], 1), s_W4) + s_b4
    
    s_W5 = tf.Variable(tf.zeros([266, 256]))
    s_b5 = tf.Variable(tf.zeros([256]))
    s_z5 = tf.matmul(tf.concat([a5, y_placeholder], 1), s_W5) + s_b5
    
    #Layer Weights Update
    W1_delta = tf.matmul(tf.transpose(X_placeholder), (activation_diff(n1, a1) * s_z1))
    W1_update = W1.assign(W1 - learning_rate * W1_delta)
    W2_delta = tf.matmul(tf.transpose(a1), (activation_diff(n2, a2) * s_z2))
    W2_update = W2.assign(W2 - learning_rate * W2_delta)
    W3_delta = tf.matmul(tf.transpose(a2), (activation_diff(n3, a3) * s_z3))
    W3_update = W3.assign(W3 - learning_rate * W3_delta)
    W4_delta = tf.matmul(tf.transpose(a3), (activation_diff(n4, a4) * s_z4))
    W4_update = W4.assign(W4 - learning_rate * W4_delta)
    W5_delta = tf.matmul(tf.transpose(a4), (activation_diff(n5, a5) * s_z5))
    W5_update = W5.assign(W5 - learning_rate * W5_delta)
    
    W6_delta = tf.matmul(tf.transpose(a5), W6_gradients)#True gradients
    W6_update = W6.assign(W6 - learning_rate * W6_delta)
    
    #synthetic Weights Update
    l5_true_gradients = tf.matmul(W6_gradients, tf.transpose(W6))
    l5_s_error = tf.reduce_mean(tf.reduce_sum(tf.square(s_z5 - l5_true_gradients), axis = 1))
    s_W5_delta = tf.reshape(tf.gradients(ys = l5_s_error, xs = s_W5), [266, 256])
    s_W5_update = s_W5.assign(s_W5 - learning_rate * s_W5_delta)
    
    l4_true_gradients = tf.matmul(s_z5, tf.transpose(W5))
    l4_s_error = tf.reduce_mean(tf.reduce_sum(tf.square(s_z4 - l4_true_gradients), axis = 1))
    s_W4_delta = tf.reshape(tf.gradients(ys = l4_s_error, xs = s_W4), [266, 256])
    s_W4_update = s_W4.assign(s_W4 - learning_rate * s_W4_delta)
    
    l3_true_gradients = tf.matmul(s_z4, tf.transpose(W4))
    l3_s_error = tf.reduce_mean(tf.reduce_sum(tf.square(s_z3 - l3_true_gradients), axis = 1))
    s_W3_delta = tf.reshape(tf.gradients(ys = l3_s_error, xs = s_W3), [266, 256])
    s_W3_update = s_W3.assign(s_W3 - learning_rate * s_W3_delta)
    
    l2_true_gradients = tf.matmul(s_z3, tf.transpose(W3))
    l2_s_error = tf.reduce_mean(tf.reduce_sum(tf.square(s_z2 - l2_true_gradients), axis = 1))
    s_W2_delta = tf.reshape(tf.gradients(ys = l2_s_error, xs = s_W2), [266, 256])
    s_W2_update = s_W2.assign(s_W2 - learning_rate * s_W2_delta)
    
    l1_true_gradients = tf.matmul(s_z2, tf.transpose(W2))
    l1_s_error = tf.reduce_mean(tf.reduce_sum(tf.square(s_z1 - l1_true_gradients), axis = 1))
    s_W1_delta = tf.reshape(tf.gradients(ys = l1_s_error, xs = s_W1), [266, 256])
    s_W1_update = s_W1.assign(s_W1 - learning_rate * s_W1_delta)
    
    #prediction
    correct_prediction = tf.equal(tf.argmax(a6,1), tf.argmax(y_placeholder,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    init = tf.global_variables_initializer()
    
    with tf.Session(config = tf.ConfigProto(allow_soft_placement = True)) as sess:
        sess.run(init)
        for iters in range(iterations):
            for batch in range(int (n / batch_size)):
                batch_xs = X_train[(batch*batch_size) : (batch+1)*batch_size]
                batch_ys = y_train[(batch*batch_size) : (batch+1)*batch_size]
                sess.run([W1_update, W2_update, W3_update, W4_update, W5_update, W6_update,\
                          s_W5_update, s_W4_update, s_W3_update, s_W2_update, s_W1_update], feed_dict = {X_placeholder: batch_xs, y_placeholder: batch_ys})
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