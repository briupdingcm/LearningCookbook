import matplotlib.pyplot as plt
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

    # Create variables for linear regression
    A = tf.Variable(tf.random_normal(shape=[1, 1]), name="weight")
    b = tf.Variable(tf.random_normal(shape=[1, 1]), name="bias")

    with g.name_scope("inference"):
        # Declare model operations
        model_output = tf.add(tf.matmul(x_data, A), b)

    with g.name_scope("loss"):
        loss = tf.reduce_mean(tf.square(y_target - model_output))

    with g.name_scope("train"):
        learning_rate = 0.1
        my_opt = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = my_opt.minimize(loss)

    loss_vec = []

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(10000):
            rand_x, rand_y = input()
            sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
            temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
            loss_vec.append(temp_loss)
            if (i + 1) % 25 == 0:
                print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
                print('Loss = ' + str(temp_loss))
        best_fit = sess.run(model_output, feed_dict={x_data: x_vals, y_target: y_vals})

    with tf.name_scope("draw"):
        plt.plot(x_vals, y_vals, 'o', label='Data Points')
        plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
        plt.legend(loc='upper left')
        plt.title('Sepal Length vs Pedal Width')
        plt.xlabel('Pedal Width')
        plt.ylabel('Sepal Length')
        plt.show()

        # Plot loss over time
        plt.plot(loss_vec, 'k-')
        plt.title('L2 Loss per Generation')
        plt.xlabel('Generation')
        plt.ylabel('L2 Loss')
        plt.show()