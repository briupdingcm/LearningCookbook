import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime as dt

input_layer_nodes = 2

output_layer_nodes = 2

first_hidden_nodes = 10

second_hidden_nodes = 10

samples_num = 30000

iteration = 100000

batch_size = 1000

learning_rate = 0.2

def error_rate(Y, Z):
    t1 = [1.0 if e1 >= e2 else 0.0 for [e1, e2] in Y]
    t2 = [1.0 if e1 >= e2 else 0.0 for [e1, e2] in Z]
    t = [1.0 if e1 == e2 else 0.0 for (e1, e2) in zip(t1, t2)]
    succ = np.sum(t)
    return succ / len(Y)


def dataset():
    x_data = np.reshape(np.random.rand(samples_num*input_layer_nodes), [samples_num, input_layer_nodes])
    y_data = [[1., 0.] if (x0 >= 0.5 and x1 >= 0.5) or (x0 <= 0.5 and x1 <= 0.5) else [0., 1.] for [x0, x1] in x_data]
    return np.array(x_data), np.array(y_data)


x, y = dataset()


def get_data():
    rand_index = np.random.choice(samples_num, size=batch_size)
    rand_x = x[rand_index]
    rand_y = y[rand_index]
    return rand_x, rand_y


def draw():
    c1 = np.array([x0 for (x0, [y0, y1]) in zip(x, y) if y0 > y1])
    c2 = np.array([x0 for (x0, [y0, y1]) in zip(x, y) if y0 <= y1])

    plt.title('Sample')

    plt.plot(c1[:, 0], c1[:, 1], 'b.')
    plt.plot(c2[:, 0], c2[:, 1], 'r.')
    plt.show()


draw()

start_time = dt.datetime.now()

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
        loss = tf.reduce_mean(tf.square(final_out - y_target))

    with tf.name_scope('train'):
        my_opt = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = my_opt.minimize(loss)

    loss_vec = []
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(iteration):
            rand_x, rand_y = get_data()

            sess.run(train_step, feed_dict={x_input: rand_x, y_target: rand_y})
            error = sess.run(loss, feed_dict={x_input: rand_x, y_target: rand_y})
            loss_vec.append(error)
            if i % 100 == 0:
                print("loss: " + str(error))
        y_predict = sess.run(final_out, feed_dict={x_input: x, y_target: y})
        end_time = dt.datetime.now()
        print(end_time - start_time)

    with tf.name_scope('draw'):
        c1 = np.array([x0 for (x0, [y0, y1]) in zip(x, y_predict) if y0 <= y1])
        c2 = np.array([x0 for (x0, [y0, y1]) in zip(x, y_predict) if y0 > y1])

        plt.title('XOR Result: ' + str(error_rate(y, y_predict)))

        plt.plot(c1[:, 0], c1[:, 1], 'b.')
        plt.plot(c2[:, 0], c2[:, 1], 'r.')
        plt.show()

        plt.plot(loss_vec, 'r.')
        plt.title('Loss (MSE) per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.show()