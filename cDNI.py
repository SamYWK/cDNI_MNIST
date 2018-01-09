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
    learning_rate = 0.001
    iterations = 100
    
    X_placeholder = tf.placeholder(tf.float32, [None, 784])
    y_placeholder = tf.placeholder(tf.float32, [None, 10])
    
    #add layers
    W1 = tf.Variable(tf.zeros([784, 100]))
    b1 = tf.Variable(tf.zeros([100]))
    z1 = tf.matmul(X_placeholder, W1) + b1
    a1 = tf.nn.sigmoid(z1)
    
    W2 = tf.Variable(tf.random_normal([100, 100]))
    b2 = tf.Variable(tf.random_normal([100]))
    z2 = tf.matmul(a1, W2) + b2
    a2 = tf.nn.sigmoid(z2)
    
    W3 = tf.Variable(tf.random_normal([100, 100]))
    b3 = tf.Variable(tf.random_normal([100]))
    z3 = tf.matmul(a2, W3) + b3
    a3 = tf.nn.sigmoid(z3)
    
    W4 = tf.Variable(tf.random_normal([100, 100]))
    b4 = tf.Variable(tf.random_normal([100]))
    z4 = tf.matmul(a3, W4) + b4
    a4 = tf.nn.sigmoid(z4)
    
    W5 = tf.Variable(tf.random_normal([100, 100]))
    b5 = tf.Variable(tf.random_normal([100]))
    z5 = tf.matmul(a4, W5) + b5
    a5 = tf.nn.sigmoid(z5)
    
    W6 = tf.Variable(tf.random_normal([100, 100]))
    b6 = tf.Variable(tf.random_normal([100]))
    z6 = tf.matmul(a5, W6) + b6
    a6 = tf.nn.sigmoid(z6)
    
    W7 = tf.Variable(tf.random_normal([100, 10]))
    b7 = tf.Variable(tf.random_normal([10]))
    z7 = tf.matmul(a6, W7) + b7
    a7 = tf.nn.softmax(z7)
    
    #loss
    soft = tf.reshape(tf.nn.softmax_cross_entropy_with_logits(labels = y_placeholder, logits = z7), [100, 1])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_placeholder, logits = z7))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    #True gradients
    W7_gradients = tf.reshape(tf.gradients(ys = cross_entropy, xs = z7), [100, 10])
    
    #synthetic gradients
    s_W1 = tf.Variable(tf.zeros([111, 100]))
    s_b1 = tf.Variable(tf.zeros([100]))
    s_z1 = tf.matmul(tf.concat([a1, soft, y_placeholder], 1), s_W1) + s_b1
    
    s_W2 = tf.Variable(tf.zeros([111, 100]))
    s_b2 = tf.Variable(tf.zeros([100]))
    s_z2 = tf.matmul(tf.concat([a2, soft, y_placeholder], 1), s_W2) + s_b2
    
    s_W3 = tf.Variable(tf.zeros([111, 100]))
    s_b3 = tf.Variable(tf.zeros([100]))
    s_z3 = tf.matmul(tf.concat([a3, soft, y_placeholder], 1), s_W3) + s_b3
    
    s_W4 = tf.Variable(tf.zeros([111, 100]))
    s_b4 = tf.Variable(tf.zeros([100]))
    s_z4 = tf.matmul(tf.concat([a4, soft, y_placeholder], 1), s_W4) + s_b4
    
    s_W5 = tf.Variable(tf.zeros([111, 100]))
    s_b5 = tf.Variable(tf.zeros([100]))
    s_z5 = tf.matmul(tf.concat([a5, soft, y_placeholder], 1), s_W5) + s_b5
    
    s_W6 = tf.Variable(tf.zeros([111, 100]))
    s_b6 = tf.Variable(tf.zeros([100]))
    s_z6 = tf.matmul(tf.concat([a6, soft, y_placeholder], 1), s_W6) + s_b6
    
    #Layer Weights Update
    W1_delta = tf.matmul(tf.transpose(X_placeholder), (a1*(1 - a1) * s_z1))
    W1_update = W1.assign(W1 - learning_rate * W1_delta)
    W2_delta = tf.matmul(tf.transpose(a1), (a2*(1 - a2) * s_z2))
    W2_update = W2.assign(W2 - learning_rate * W2_delta)
    W3_delta = tf.matmul(tf.transpose(a2), (a3*(1 - a3) * s_z3))
    W3_update = W3.assign(W3 - learning_rate * W3_delta)
    W4_delta = tf.matmul(tf.transpose(a3), (a4*(1 - a4) * s_z4))
    W4_update = W4.assign(W4 - learning_rate * W4_delta)
    W5_delta = tf.matmul(tf.transpose(a4), (a5*(1 - a5) * s_z5))
    W5_update = W5.assign(W5 - learning_rate * W5_delta)
    W6_delta = tf.matmul(tf.transpose(a5), (a6*(1 - a6) * s_z6))
    W6_update = W6.assign(W6 - learning_rate * W6_delta)
    
    W7_delta = tf.matmul(tf.transpose(a6), W7_gradients)#True gradients
    W7_update = W7.assign(W7 - learning_rate * W7_delta)
    
    #synthetic Weights Update
    l6_true_gradients = tf.matmul(W7_gradients, tf.transpose(W7))
    l6_s_error = tf.reduce_mean(tf.reduce_sum(tf.square(s_z6 - l6_true_gradients), axis = 1))
    s_W6_delta = tf.reshape(tf.gradients(ys = l6_s_error, xs = s_W6), [111, 100])
    s_W6_update = s_W6.assign(s_W6 - learning_rate * s_W6_delta)
    
    l5_true_gradients = tf.matmul(s_z6, tf.transpose(W6))
    l5_s_error = tf.reduce_mean(tf.reduce_sum(tf.square(s_z5 - l5_true_gradients), axis = 1))
    s_W5_delta = tf.reshape(tf.gradients(ys = l5_s_error, xs = s_W5), [111, 100])
    s_W5_update = s_W5.assign(s_W5 - learning_rate * s_W5_delta)
    
    l4_true_gradients = tf.matmul(s_z5, tf.transpose(W5))
    l4_s_error = tf.reduce_mean(tf.reduce_sum(tf.square(s_z4 - l4_true_gradients), axis = 1))
    s_W4_delta = tf.reshape(tf.gradients(ys = l4_s_error, xs = s_W4), [111, 100])
    s_W4_update = s_W4.assign(s_W4 - learning_rate * s_W4_delta)
    
    l3_true_gradients = tf.matmul(s_z4, tf.transpose(W4))
    l3_s_error = tf.reduce_mean(tf.reduce_sum(tf.square(s_z3 - l3_true_gradients), axis = 1))
    s_W3_delta = tf.reshape(tf.gradients(ys = l3_s_error, xs = s_W3), [111, 100])
    s_W3_update = s_W3.assign(s_W3 - learning_rate * s_W3_delta)
    
    l2_true_gradients = tf.matmul(s_z3, tf.transpose(W3))
    l2_s_error = tf.reduce_mean(tf.reduce_sum(tf.square(s_z2 - l2_true_gradients), axis = 1))
    s_W2_delta = tf.reshape(tf.gradients(ys = l2_s_error, xs = s_W2), [111, 100])
    s_W2_update = s_W2.assign(s_W2 - learning_rate * s_W2_delta)
    
    l1_true_gradients = tf.matmul(s_z2, tf.transpose(W2))
    l1_s_error = tf.reduce_mean(tf.reduce_sum(tf.square(s_z1 - l1_true_gradients), axis = 1))
    s_W1_delta = tf.reshape(tf.gradients(ys = l1_s_error, xs = s_W1), [111, 100])
    s_W1_update = s_W1.assign(s_W1 - learning_rate * s_W1_delta)
    
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
                sess.run([W1_update, W2_update, W3_update, W4_update, W5_update, W6_update, W7_update,\
                          s_W6_update, s_W5_update, s_W4_update, s_W3_update, s_W2_update, s_W1_update], feed_dict = {X_placeholder: batch_xs, y_placeholder: batch_ys})
                if batch % 10 == 0:
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