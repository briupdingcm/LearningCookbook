import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf



x = [
    [0., 0.],
    [1., 0.],
    [0., 1.],
    [1., 1.],
]
y = [[1], [0], [0], [1]]

input_layer_nodes = 2

output_layer_nodes = 1

first_hidden_nodes = 6

second_hidden_nodes = 3





g = tf.Graph()

with g.as_default():
    w1 = tf.Variable(tf.random_normal(shape=[input_layer_nodes, first_hidden_nodes], dtype=tf.float32))
    b1 = tf.Variable(tf.random_normal(shape=[first_hidden_nodes], dtype=tf.float32))

    w2 = tf.Variable(tf.random_normal(shape=[first_hidden_nodes, second_hidden_nodes], dtype=tf.float32))
    b2 = tf.Variable(tf.random_normal(shape=[second_hidden_nodes], dtype=tf.float32))

    w3 = tf.Variable(tf.random_normal(shape=[second_hidden_nodes, output_layer_nodes], dtype=tf.float32))
    b3 = tf.Variable(tf.random_normal(shape=[output_layer_nodes], dtype=tf.float32))

    with tf.name_scope('input_layer'):
        x_input = tf.placeholder(shape=[None, input_layer_nodes], dtype=tf.float32)
        y_target = tf.placeholder(shape=[None, output_layer_nodes], dtype=tf.float32)

    with tf.name_scope('first_hidden_layer'):
        first_out = tf.sigmoid(tf.add(tf.matmul(x_input, w1), b1))

    with tf.name_scope('second_hidden_layer'):
        second_out = tf.sigmoid(tf.add(tf.matmul(first_out, w2), b2))

    with tf.name_scope('output_layer'):
        final_out = tf.sigmoid(tf.add(tf.matmul(second_out, w3), b3))

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(first_out - y_target))

    with tf.name_scope('train'):
        my_opt = tf.train.GradientDescentOptimizer(0.01)
        train_step = my_opt.minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for i in range(10000):
            sess.run(train_step, feed_dict={x_input: x, y_target: y})
            print('i = ' + str(i) + ' ' )
            print(sess.run(loss, feed_dict={x_input: x, y_target: y}))
            print(sess.run(final_out, feed_dict={x_input: x, y_target: y}))