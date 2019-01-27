import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D


num_points = 1000
vectors_set = []

def sigmoid(x):
    return np.divide(1.0, (1.0 + np.exp(0.0 - x)))

def datasets():
    x = np.transpose(np.matrix(np.random.normal(0.0, 0.5, num_points)))
    y = np.transpose(np.matrix(np.random.normal(0.0, 0.3, num_points)))
    t = np.transpose(np.matrix(np.repeat(1.0, len(x))))
    l = sigmoid(0.98 * x + 0.07)
    t[y < l] = 0
    print(t)
    return x, y, l

x_vals, y_vals, t = datasets()
X, Y = np.meshgrid(x_vals, y_vals)
figure = plt.figure(1, figsize = (12, 4))
subplot3d = plt.subplot(111, projection='3d')
surface = subplot3d.plot_surface(x_vals, y_vals, t, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, linewidth=0.1)
plt.show()

plt.contourf(X, Y, t, cmap=matplotlib.cm.coolwarm)
plt.colorbar()
plt.show()

learning_rate = 0.3

loss_vec = []
g = tf.Graph()


def inputs():
    x_vals_column = np.transpose(np.matrix(x_vals))
    ones_column = np.transpose(np.matrix(np.repeat(1, num_points)))
    A = np.column_stack((x_vals, ones_column))
    b = np.transpose(np.matrix(y_vals))
    return A, np.divide(1.0, (1.0 + np.exp(0.0 - b)))


A, b = inputs()
plt.plot(x_vals, b, 'r.')
plt.show()

def get_batch(size=1):
    rand_idx = np.random.choice(len(A), size)
    x_rand_val = A[rand_idx]
    y_rand_val = b[rand_idx]
    return x_rand_val, y_rand_val


with g.as_default():
    w_data = tf.Variable(tf.random_normal((2, 1)))

    with g.name_scope('input') as scope:
        x_data = tf.placeholder(tf.float32, shape=(None, 2), name="input_x")
        y_data = tf.placeholder(tf.float32, shape=(None, 1), name="input_y")

    with g.name_scope('output') as scope:
        y_pred = tf.sigmoid(tf.matmul(x_data, w_data))

    with g.name_scope('loss') as scope:
    #    diff = tf.subtract(y_data, y_pred)
     #   loss = tf.reduce_mean(tf.reduce_sum(tf.pow(diff, 2.0), axis=1))
        loss = tf.reduce_mean(tf.square(y_data - y_pred))

    with g.name_scope('train') as scope:
        #w1 = tf.subtract(y_data, y_pred)
        #w2 = tf.subtract(1.0, y_pred)
        #w3 = tf.multiply(w1, w2)
        #w4 = tf.multiply(w3, y_pred)
        #w5 = tf.multiply(w4, x_data)
        #w6 = tf.transpose(w5)
        #grad_w = w6
        #update_w = w_data.assign(w_data - learning_rate * grad_w)
        #updates = tf.group(update_w)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(10000):
            x_rand_val, y_rand_val = get_batch()
            loss_val, _ = sess.run([loss, train_step], feed_dict={x_data: x_rand_val, y_data: y_rand_val})
            loss_vec.append(loss_val)
        y_pred_val = sess.run(y_pred, feed_dict={x_data: A})
        w_val = sess.run(w_data)
    print(w_val)
    #plt.plot(x_vals, y_pred_val)
    plt.plot(y_vals, b, 'r.')
    plt.show()