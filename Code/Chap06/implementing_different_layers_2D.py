# The layers of interest are:
#  (1) Convolutional Layer
#  (2) Activation Layer
#  (3) Max-Pool Layer
#  (4) Fully Connected Layer
#

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

data_size = [10, 10]


def input():
    return np.random.normal(size=data_size)


data_2d = input()

def conv_layer_2d(input_2d, my_filter):
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    convolution_output = tf.nn.conv2d(input_4d, filter=my_filter, strides=[1, 2, 2, 1], padding='VALID')
    conv_output_2d = tf.squeeze(convolution_output)
    return conv_output_2d


def max_pool(input_2d, width, height):
    input_3d = tf.expand_dims(input_2d, 0)
    input_4d = tf.expand_dims(input_3d, 3)
    pool_output = tf.nn.max_pool(input_4d, ksize=[1, height, width, 1], strides=[1,1,1,1], padding='VALID')
    return tf.squeeze(pool_output)


def activation(input_2d):
    return tf.nn.relu(input_2d)


def fully_connected(input_layer, num_outputs):
    flat_input = tf.reshape(input_layer, [-1])
    weight_shape = [tf.shape(flat_input)[0], num_outputs]#tf.squeeze([])
    weight = tf.random_normal(weight_shape, stddev=0.1)
    bias = tf.random_normal(shape=[num_outputs])
    input_2d = tf.expand_dims(flat_input, 0)
    full_output = tf.add(tf.matmul(input_2d, weight), bias)
    return tf.squeeze(full_output)

g = tf.Graph()

with g.as_default():
    my_filter = tf.Variable(tf.random_normal(shape=[2, 2, 1, 1]))

    with tf.name_scope('input_layer'):
        x_input_2d = tf.placeholder(dtype=tf.float32, shape=data_size)

    with tf.name_scope('convolution_layer'):
        my_convolution_output = conv_layer_2d(x_input_2d, my_filter)
        my_activation_output = activation(my_convolution_output)

    with tf.name_scope('pool_layer'):
        my_maxpool_output = max_pool(my_activation_output, width=2, height=2)

    with tf.name_scope('fully_connect_layer'):
        my_full_output = fully_connected(my_maxpool_output, 5)

    feed_dict = {x_input_2d: data_2d}

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        print('filter ')
        print(sess.run(my_filter))

        # Convolution Output
        print('Input = [10 X 10] array')
        print('2x2 Convolution, stride size = [2x2], results in the [5x5] array:')
        print(sess.run(my_convolution_output, feed_dict=feed_dict))

        # Activation Output
        print('\nInput = the above [5x5] array')
        print('ReLU element wise returns the [5x5] array:')
        print(sess.run(my_activation_output, feed_dict=feed_dict))

        # Max Pool Output
        print('\nInput = the above [5x5] array')
        print('MaxPool, stride size = [1x1], results in the [4x4] array:')
        print(sess.run(my_maxpool_output, feed_dict=feed_dict))

        # Fully Connected Output
        print('\nInput = the above [4x4] array')
        print('Fully connected layer on all four rows with five outputs:')
        print(sess.run(my_full_output, feed_dict=feed_dict))