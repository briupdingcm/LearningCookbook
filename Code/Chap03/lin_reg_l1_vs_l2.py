import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

def load_data():
    iris = datasets.load_iris()
    x_vals = [x[3] for x in iris.data]
    y_vals = np.array([y[0] for y in iris.data])
    return np.transpose(np.matrix(x_vals)), np.transpose(np.matrix(y_vals))

x_vals, y_vals = load_data()

batch_size = 5

def input():
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = x_vals[rand_index]
    rand_y = y_vals[rand_index]
    return rand_x, rand_y

iteration = 500

g = tf.Graph()
with g.as_default():
    x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    A = tf.Variable(tf.random_normal(shape=[1,1]))
    b = tf.Variable(tf.random_normal(shape=[1,1]))

    loss_vec_l1 = []
    with g.name_scope("inference"):
        model_output = tf.add(tf.matmul(x_data, A), b)

    with g.name_scope("loss"):
        loss_l1 = tf.reduce_mean(tf.abs(model_output-y_target))

    with g.name_scope("train"):
        learning_rate = 0.1
        my_opt_l1 = tf.train.GradientDescentOptimizer(learning_rate)
        train_step_l1 = my_opt_l1.minimize(loss_l1)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(iteration):
            rand_x, rand_y = input()
            sess.run(train_step_l1, feed_dict={x_data: rand_x, y_target: rand_y})
            temp_loss_l1 = sess.run(loss_l1, feed_dict={x_data: x_vals, y_target: y_vals})
            loss_vec_l1.append(temp_loss_l1)
            if (i + 1) % 25 == 0:
                print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))

    with g.name_scope("draw"):
        plt.plot(loss_vec_l1, 'k-')
        plt.show()