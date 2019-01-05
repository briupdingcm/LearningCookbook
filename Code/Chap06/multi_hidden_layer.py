import matplotlib.pyplot as plt
import my_data as md
import numpy as np
import tensorflow as tf

DATA_DIR = '/Users/kevinding/data/ml/MNIST'
NUM_STEPS = 1000
MINIBATCH_SIZE = 100
learning_rate = 1e-4

mnist = md.read_data_sets(DATA_DIR, one_hot=True)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1.0)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)


def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.add(tf.matmul(input, W), b)


with tf.Session() as sess:
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)

    with tf.name_scope('input_layer'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    with tf.name_scope('convolution_layer_1'):
        conv1 = conv_layer(x_image, shape=[5, 5, 1, 32])

    with tf.name_scope('pool_layer_1'):
        conv1_pool = max_pool_2x2(conv1)

    with tf.name_scope('convolution_layer_2'):
        conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])

    with tf.name_scope('pool_layer_2'):
        conv2_pool = max_pool_2x2(conv2)

    with tf.name_scope('full_layer_1'):
        conv2_flat = tf.reshape(conv2_pool, [-1, 7*7*64])
        full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

    with tf.name_scope('full_layer_2'):
        full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)
        y_conv = full_layer(full1_drop, 10)

    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train_accuracy_list = []
    test_accuracy_list = []


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(NUM_STEPS):
            batch = mnist.train.next_batch(MINIBATCH_SIZE)

            if i % 100 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print("step {}, training accuracy {}".format(i, train_accuracy))
                train_accuracy_list.append(train_accuracy)

                sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        X = mnist.test.images.reshape(10, 1000, 784)
        Y = mnist.test.labels.reshape(10, 1000, 10)
        test_accuracy = np.mean([sess.run(accuracy, feed_dict={x:X[i], y_:Y[i], keep_prob: 1.0}) for i in range(10)])

    print("test accuracy: {}".format(test_accuracy))

    with tf.name_scope('draw'):
        # Plot the activation values
        plt.plot(train_accuracy_list, 'r-', label='train accuracy')
        plt.plot(test_accuracy_list, 'b-', label='test accuracy')
        plt.ylim([0, 1.0])
        plt.title('Accuracy Outputs')
        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper right')
        plt.show()