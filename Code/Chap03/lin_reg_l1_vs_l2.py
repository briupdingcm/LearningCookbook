import matplotlib as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

def load_data():
    iris = datasets.load_iris()
    x_vals = np.array([x[3] for x in iris.data])
    y_vals = np.array([y[0] for y in iris.data])
    return np.transpose(np.matrix(x_vals)), np.transpose(np.matrix(y_vals))

x_vals, y_vals = load_data()

batch_size = 5

def input():
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = x_vals[rand_index]
    rand_y = y_vals[rand_index]
    return rand_x, rand_y


g = tf.Graph()
with g.as_default():
    x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    A = tf.Variable(tf.random_normal(shape=[1,1]))
    b = tf.Variable(tf.random_normal(shape=[1,1]))

