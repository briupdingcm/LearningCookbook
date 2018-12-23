import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

def load_data():
    iris = datasets.load_iris()
    x = np.array([x[3] for x in iris.data])
    y = np.array([y[0] for y in iris.data])
    return np.transpose(np.matrix(x, dtype=np.float32)), np.transpose(np.matrix(y, dtype=np.float32))


x_vals, y_vals = load_data()

batch_size = 50

def input():
    rand_idx = np.random.choice(len(x_vals), size=batch_size)
    rand_x = x_vals[rand_idx]
    rand_y = y_vals[rand_idx]
    return rand_x, rand_y


g = tf.Graph()
learning_rate = 0.1
iteration = 25000
with g.as_default():
    x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    A = tf.Variable(tf.random_normal(shape=[1,1]))
    b = tf.Variable(tf.random_normal(shape=[1,1]))

    with tf.name_scope("inference"):
        model_output = tf.add(tf.matmul(x_data, A), b)

    with tf.name_scope("loss"):
        demming_numerator = tf.abs(tf.subtract(y_target, tf.add(tf.matmul(x_data, A), b)))
        demming_denominator = tf.sqrt(tf.add(tf.square(A), 1))
        loss = tf.reduce_mean(tf.truediv(demming_numerator, demming_denominator))

    with tf.name_scope("train"):
        my_opt = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = my_opt.minimize(loss)

    loss_vec = []
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(iteration):
            rand_x, rand_y = input()
            sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
            temp_loss = sess.run(loss, feed_dict={x_data: x_vals, y_target: y_vals})
            loss_vec.append(temp_loss)
            if (i + 1) % 50 == 0:
                print('Step #''' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
                print('loss = ' + str(temp_loss))
        best_fit = sess.run(model_output, feed_dict={x_data: x_vals, y_target: y_vals})
        writer = tf.summary.FileWriter('./demming_graph', g)

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
