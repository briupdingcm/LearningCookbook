import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.set_random_seed(5)
np.random.seed(42)

g = tf.Graph()

def load_dataset():
    return np.random.normal(2, 0.1, 500)


x = load_dataset()
batch_size = 50


def input():
    rand_indices = np.random.choice(len(x), size=batch_size)
    x_vals = np.transpose([x[rand_indices]])
    return x_vals


with g.as_default():
    a1 = tf.Variable(tf.random_normal(shape=[1, 1]))
    b1 = tf.Variable(tf.random_uniform(shape=[1, 1]))
    a2 = tf.Variable(tf.random_normal(shape=[1, 1]))
    b2 = tf.Variable(tf.random_uniform(shape=[1, 1]))

    x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    with g.name_scope('activation'):
        sigmoid_activation = tf.sigmoid(tf.add(tf.matmul(x_data, a1), b1))
        relu_activation = tf.nn.relu(tf.add(tf.matmul(x_data, a2), b2))

    with g.name_scope('loss'):
        loss1 = tf.reduce_mean(tf.square(tf.subtract(sigmoid_activation, 0.75)))
        loss2 = tf.reduce_mean(tf.square(tf.subtract(relu_activation, 0.75)))

    with g.name_scope('train'):
        my_opt = tf.train.GradientDescentOptimizer(0.01)
        train_step_sigmoid = my_opt.minimize(loss1)
        train_step_relu = my_opt.minimize(loss2)

    loss_vec_sigmoid = []
    loss_vec_relu = []
    activation_sigmoid = []
    activation_relu = []

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(750):
            x_vals = input()
            sess.run(train_step_sigmoid, feed_dict={x_data: x_vals})
            sess.run(train_step_relu, feed_dict={x_data: x_vals})

            loss_vec_sigmoid.append(sess.run(loss1, feed_dict={x_data: x_vals}))
            loss_vec_relu.append(sess.run(loss2, feed_dict={x_data: x_vals}))

            activation_sigmoid.append(np.mean(sess.run(sigmoid_activation, feed_dict={x_data: x_vals})))
            activation_relu.append(np.mean(sess.run(relu_activation, feed_dict={x_data: x_vals})))

    with tf.name_scope('draw'):
        # Plot the activation values
        plt.plot(activation_sigmoid, 'k-', label='Sigmoid Activation')
        plt.plot(activation_relu, 'r--', label='Relu Activation')
        plt.ylim([0, 1.0])
        plt.title('Activation Outputs')
        plt.xlabel('Generation')
        plt.ylabel('Outputs')
        plt.legend(loc='upper right')
        plt.show()

        # Plot the loss
        plt.plot(loss_vec_sigmoid, 'k-', label='Sigmoid Loss')
        plt.plot(loss_vec_relu, 'r--', label='Relu Loss')
        plt.ylim([0, 1.0])
        plt.title('Loss per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.show()