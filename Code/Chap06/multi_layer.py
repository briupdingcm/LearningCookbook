import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime as dt

input_layer_nodes = 2

output_layer_nodes = 2

first_hidden_nodes = 10

second_hidden_nodes = 10

third_hidden_nodes = 8

samples_num = 30

iteration = 10000

batch_size = 1000

learning_rate = 0.2


def dataset():
    x_data = np.reshape(np.random.rand(samples_num*input_layer_nodes), [samples_num, input_layer_nodes])
    y_data = [[1., 0.] if (x0 >= 0.5 and x1 >= 0.5) or (x0 <= 0.5 and x1 <= 0.5) else [0., 1.] for [x0, x1] in x_data]
    return np.array(x_data), np.array(y_data)


x, y = dataset()


def get_data():
    rand_index = np.random.choice(samples_num, size=batch_size)

    return x[rand_index], y[rand_index]#rand_x, rand_y


def draw(X, Y, title):
   # test = np.argmax(Y, 1)
    c1 = X[Y == 1]
    c2 = X[Y == 0]
   # c1 = np.array([x0 for (x0, [y0, y1]) in zip(X, Y) if y0 > y1])
   # c2 = np.array([x0 for (x0, [y0, y1]) in zip(X, Y) if y0 <= y1])

    plt.title(title)

    plt.plot(c1[:, 0], c1[:, 1], 'b.')
    plt.plot(c2[:, 0], c2[:, 1], 'r.')
    plt.show()


draw(x, np.argmax(y, 1), 'Sample')

start_time = dt.datetime.now()

g = tf.Graph()


def init_weight(shape, st_dev):
    weight = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return weight


def init_bais(shape, st_dev):
    bias = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return bias


def fully_connect(input_layer, weights, biases):
    layer = tf.add(tf.matmul(input_layer, weights), biases)
    return tf.nn.sigmoid(layer)


with g.as_default():
    w1 = init_weight(shape=[input_layer_nodes, first_hidden_nodes], st_dev=1.0)
    b1 = init_bais(shape=[first_hidden_nodes], st_dev=1.0)

    w2 = init_weight(shape=[first_hidden_nodes, second_hidden_nodes], st_dev=1.0)
    b2 = init_bais(shape=[second_hidden_nodes], st_dev=1.0)

    w3 = init_weight(shape=[second_hidden_nodes, third_hidden_nodes], st_dev=1.0)
    b3 = init_bais(shape=[third_hidden_nodes], st_dev=1.0)

    w4 = init_weight(shape=[third_hidden_nodes, output_layer_nodes], st_dev=1.0)
    b4 = init_bais(shape=[output_layer_nodes], st_dev=1.0)

    with tf.name_scope('input_layer'):
        x_input = tf.placeholder(shape=[None, input_layer_nodes], dtype=tf.float32)
        y_target = tf.placeholder(shape=[None, output_layer_nodes], dtype=tf.float32)

    with tf.name_scope('first_hidden_layer'):
        first_out = fully_connect(x_input, w1, b1)

    with tf.name_scope('second_hidden_layer'):
        second_out = fully_connect(first_out, w2, b2)

    with tf.name_scope('third_hidden_layer'):
        third_out = fully_connect(second_out, w3, b3)

    with tf.name_scope('output_layer'):
        final_out = fully_connect(third_out, w4, b4)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(final_out - y_target))

    with tf.name_scope('train'):
        my_opt = tf.train.AdamOptimizer(learning_rate)
        train_step = my_opt.minimize(loss)

    with tf.name_scope('result'):
        output_predict = tf.cast(tf.argmax(final_out, 1), tf.float32)
        output_target = tf.cast(tf.argmax(y_target, 1), tf.float32)
        correct_mask = tf.equal(output_target, output_predict)#tf.argmax(final_out, 1), tf.argmax(y_target, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

    loss_vec = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(iteration):
            rand_x, rand_y = get_data()

            sess.run(train_step, feed_dict={x_input: rand_x, y_target: rand_y})
            train_loss = sess.run(loss, feed_dict={x_input: rand_x, y_target: rand_y})
            loss_vec.append(train_loss)

            if i % 1000 == 0:
                print("loss: " + str(train_loss))

        y_predict = np.argmax(sess.run(final_out, feed_dict={x_input: x, y_target: y}), 1)

        train_accuracy = sess.run(accuracy, feed_dict={x_input: x, y_target: y})
        print("accuracy: {}".format(train_accuracy))

    end_time = dt.datetime.now()
    print(end_time - start_time)

    with tf.name_scope('draw'):
        draw(x, y_predict, 'XOR Result: {} '.format(train_accuracy))

        plt.plot(loss_vec, 'r.')
        plt.title('Loss (MSE) per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.show()
