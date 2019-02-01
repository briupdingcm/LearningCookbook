import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()

x_vals = np.transpose(np.array([x[3] for x in iris.data]))
y_vals = np.transpose(np.array([y[0] for y in iris.data]))

train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

x_vals_train = np.transpose(np.matrix(x_vals[train_indices]))
y_vals_train = np.transpose(np.matrix(y_vals[train_indices]))
x_vals_test = np.transpose(np.matrix(x_vals[test_indices]))
y_vals_test = np.transpose(np.matrix(y_vals[test_indices]))

def get_data(x, y, size=10):
    rand_index = np.random.choice(len(x), size=size)
    return x[rand_index], y[rand_index]

batch_size = 50

g = tf.Graph()

with g.as_default():
    A = tf.Variable(tf.random_normal(shape=[1, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))
    epsilon = tf.constant([0.5])

    with tf.name_scope('input') as scope:
        x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    with tf.name_scope('inference') as scope:
        model_output = tf.add(tf.matmul(x_data, A), b)
        upper_output = tf.add(model_output, epsilon)
        lower_output = tf.subtract(model_output, epsilon)

    with tf.name_scope('loss') as scope:
        loss = tf.reduce_mean(tf.maximum(0., tf.subtract(tf.abs(tf.subtract(model_output, y_target)), epsilon)))

    with tf.name_scope('train') as scope:
        my_opt = tf.train.GradientDescentOptimizer(0.075)
        train_step = my_opt.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        train_loss=[]
        test_loss=[]
        for i in range(200):
            rand_x, rand_y = get_data(x_vals_train, y_vals_train, size=batch_size)
            sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

            temp_train_loss = sess.run(loss, feed_dict={x_data: x_vals_train, y_target: y_vals_train})
            train_loss.append(temp_train_loss)

            temp_test_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: y_vals_test})
            test_loss.append(temp_test_loss)
        y_pred_val, upper_val, lower_val = sess.run([model_output, upper_output, lower_output], feed_dict={x_data: np.transpose(np.matrix(x_vals))})

    with tf.name_scope('draw') as scope:
        plt.plot(train_loss, 'k-')
        plt.plot(test_loss, 'r-')
        plt.title('L2 Loss per Generation')
        plt.xlabel('Generation')
        plt.ylabel('L2 Loss')
        plt.legend(loc = 'upper right')
        plt.show()

        plt.plot(x_vals, y_vals, 'o', label='Data Points')
        plt.plot(x_vals, y_pred_val, 'r-', label='Regression Line', linewidth=3)
        plt.plot(x_vals, upper_val, 'r--', linewidth=2)
        plt.plot(x_vals, lower_val, 'r--', linewidth=2)
        plt.ylim([0, 10])
        plt.title('Sepal Length vs Pedal Width')
        plt.legend(loc='lower right')
        plt.xlabel('Pedal Width')
        plt.ylabel('Sepal Length')
        plt.show()
