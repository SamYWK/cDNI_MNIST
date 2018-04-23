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
    X = df.drop(['label'], axis = 1).values.astype(np.float32)
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
    epochs = 50
    
    g_1 = tf.Graph()
    
    with g_1.as_default():
        with tf.device('/device:GPU:0'):
            X_placeholder = tf.placeholder(tf.float32, [None, 784])
            y_placeholder = tf.placeholder(tf.float32, [None, 10])
            
            a1 = tf.layers.dense(X_placeholder, 256, tf.nn.sigmoid, name = 'layer_1')
            a2 = tf.layers.dense(a1, 256, tf.nn.sigmoid, name = 'layer_2')
            a3 = tf.layers.dense(a2, 256, tf.nn.sigmoid, name = 'layer_3')
            a4 = tf.layers.dense(a3, 256, tf.nn.sigmoid, name = 'layer_4')
            a5 = tf.layers.dense(a4, 256, tf.nn.sigmoid, name = 'layer_5')
            a6 = tf.layers.dense(a5, 10, tf.nn.sigmoid, name = 'layer_6')
            
            #loss
            loss = tf.losses.mean_squared_error(labels = y_placeholder, predictions = a6)
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            
            #prediction
            correct_prediction = tf.equal(tf.argmax(a6,1), tf.argmax(y_placeholder,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
            init = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement = False, log_device_placement = True)
        config.gpu_options.allow_growth = True
        
        with tf.Session(config = config) as sess:
            sess.run(init)
            for iters in range(epochs):
                for batch in range(int (n / batch_size)):
                    batch_xs = X_train[(batch*batch_size) : (batch+1)*batch_size]
                    batch_ys = y_train[(batch*batch_size) : (batch+1)*batch_size]
                    sess.run(train_step, feed_dict = {X_placeholder: batch_xs, y_placeholder: batch_ys})
                    if batch % 500 == 0:
                        print(sess.run(loss, feed_dict = {X_placeholder: batch_xs, y_placeholder: batch_ys}))
                        
            print('Accuracy :', sess.run(accuracy, feed_dict={X_placeholder: X_test, y_placeholder: y_test}))
    np.savetxt('feed_forward_NN_numbers.csv', numbers, delimiter=',')
    
def main():
    X_train, X_test, y_train, y_test = load_data('mnist_train.csv')
    start_time = time.time()
    cDNI(X_train, X_test, y_train, y_test)
    end_time = time.time()
    
    cost_time = end_time - start_time
    print ("It cost %f sec" % cost_time)
if __name__ == "__main__":
    main() 