import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

def load_data():
    iris = datasets.load_iris()
    x = np.array([[x[1], x[2], x[3]] for x in iris.data])
    y = np.array([y[0] for y in iris.data])
    return np.matrix(x, dtype=np.float32), np.transpose(np.matrix(y, dtype=np.float32))

x_vals, y_vals=load_data()

batch_size = 50
def input():
    rand_idx=np.random.choice(len(x_vals), size=batch_size)
    rand_x = x_vals[rand_idx]
    rand_y = y_vals[rand_idx]
    return rand_x, rand_y

g=tf.Graph()
learning_rate = 0.001
iteration = 1000
with g.as_default():
    x_data = tf.placeholder(shape=[None, 3], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    A = tf.Variable(tf.random_normal(shape=[3, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))

    with g.name_scope("inference"):
        model_output=tf.add(tf.matmul(x_data, A), b)

    with g.name_scope("loss"):
        elastic_param1 = tf.constant(1.)
        elastic_param2 = tf.constant(1.)
        l1_a_loss = tf.reduce_mean(tf.abs(A))
        l2_a_loss = tf.reduce_mean(tf.square(A))
        e1_term = tf.multiply(elastic_param1, l1_a_loss)
        e2_term = tf.multiply(elastic_param2, l2_a_loss)
        loss = tf.expand_dims(tf.add(tf.add(tf.reduce_mean(tf.square(y_target - model_output)), e1_term), e2_term), 0)

    with g.name_scope("train"):
        my_opt = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = my_opt.minimize(loss)

    loss_vec=[]
    with tf.Session() as sess:
        init = tf.global_variables_initializer();
        sess.run(init)
        for i in range(iteration):
            rand_x, rand_y = input()
            sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
            temp_loss = sess.run(loss, feed_dict={x_data: x_vals, y_target: y_vals})
            loss_vec.append(temp_loss[0])
            if (i + 1) % 50 == 0:
                print('Step #''' + str(i + 1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
                print('loss = ' + str(temp_loss))

    with tf.name_scope("draw"):
        # Plot loss over time
        plt.plot(loss_vec, 'k-')
        plt.title('L2 Loss per Generation')
        plt.xlabel('Generation')
        plt.ylabel('L2 Loss')
        plt.show()