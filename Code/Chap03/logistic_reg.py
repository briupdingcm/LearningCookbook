import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import  datasets
from sklearn.preprocessing import normalize
import requests

def load_dataset():
    birthdata_url = 'http://faculty.washington.edu/heagerty/Courses/b513/WEB2002/datasets/lowbwt.dat' #''https://www.umass.edu/statdata/statdata/data/lowbwt.dat'
    birth_file = requests.get(birthdata_url)
    birth_data = birth_file.text.split('\r\n')[5:]
    birth_header = [x for x in birth_data[0].split(' ') if len(x) >= 1]
    birth_data = [[float(x) for x in y.split(' ') if len(x) >= 1] for y in birth_data[1:] if len(y) >= 1]
    y_vals = np.array(x[1] for x in birth_data)
    x_vals = np.array(x[2:9] for x in birth_data)
    return x_vals, y_vals

x_vals, y_vals=load_dataset()

def input():
    train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
    test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
    x_vals_train = x_vals(train_indices)
    x_vals_test = x_vals(test_indices)
    y_vals_train = y_vals(train_indices)
    y_vals_test = y_vals(test_indices)
    return x_vals_train, y_vals_train, x_vals_test, y_vals_test

x_vals_train, y_vals_train, x_vals_test, y_vals_test = input()

def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m - col_min) / (col_max - col_min)

x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

batch_size = 25

g = tf.Graph()

with g.as_default():
    x_data = tf.placeholder(shape=[None, 7], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    A = tf.Variable(tf.random_normal(shape=[7, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))

    with tf.name_scope("inference"):
        model_output = tf.add(tf.matmul(x_data, A), b)


    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(model_output), y_target)

    with tf.name_scope("train"):
        my_opt = tf.train.GradientDescentOptimizer(0.01)
        train_step = my_opt.minimize(loss)


    with tf.name_scope("accuracy"):
        prediction = tf.round(tf.sigmoid(model_output))
        prediction_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
        accuracy = tf.reduce_mean(prediction_correct)


    loss_vec = []
    train_acc = []
    test_acc = []
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(1000):
            rand_index = np.random.choice(len(x_vals), size=batch_size)
            rand_x = x_vals_train[rand_index]
            rand_y = y_vals_train[rand_index]
            sess.run(train_step, feed_dict={x_data:rand_x, y_target:rand_y})
            temp_loss = sess.run(loss, feed_dict={x_data:rand_x, y_target:rand_y})
            loss_vec.append(temp_loss)
            temp_acc_train = sess.run(accuracy, feed_dict={x_data: x_vals_train, y_target: y_vals_train})
            train_acc.append(temp_acc_train)
            temp_acc_test = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: y_vals_test})
            test_acc.append(temp_acc_test)

