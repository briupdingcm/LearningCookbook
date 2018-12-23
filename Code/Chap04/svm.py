import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sklearn.datasets as datasets

iris = datasets.load_iris()


def load_dataset(indices):
    return iris.data[indices], iris.target[indices]


def split():
    size = len(np.array([x[0] for x in iris.data]))
    train_indices = np.random.choice(size, round(size * 0.8), replace=False)
    test_indices = np.array(list(set(range(size)) - set(train_indices)))
    return train_indices, test_indices


train_indices, test_indices = split()


#train_data, train_target = load_dataset(train_indices);
#test_data, test_target = load_dataset(test_indices)

def input(indices):
    data, target = load_dataset(indices)
    x_vals = np.array([[x[0], x[1]] for x in data])
    y_vals = np.array([1 if y == 0 else -1 for y in target])
    return np.matrix(x_vals), np.transpose(np.matrix(y_vals))


x_val_train, y_val_train = input(train_indices)
x_val_test, y_val_test = input(test_indices)

batch_size = 50

g = tf.Graph()

with g.as_default():
    x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    A = tf.Variable(tf.random_normal(shape=[2,1]))
    b = tf.Variable(tf.random_normal(shape=[1,1]))

    with tf.name_scope('inference'):
        model_output = tf.subtract(tf.matmul(x_data, A), b)

    with tf.name_scope('loss'):
        l2_norm = tf.reduce_mean(tf.square(A))
        alpha = tf.constant([0.1])
        classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))
        loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))


    with tf.name_scope('accuracy'):
        prediction = tf.sign(model_output)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))


    with tf.name_scope('train'):
        my_opt = tf.train.GradientDescentOptimizer(0.01)
        train_step = my_opt.minimize(loss)

    loss_vec = []
    train_accuracy = []
    test_accuracy = []
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1000):
            rand_index = np.random.choice(len(x_val_train), size=batch_size)
            rand_x = x_val_train[rand_index]
            rand_y = y_val_train[rand_index]
            sess.run(train_step, feed_dict={x_data:rand_x, y_target:rand_y})
            temp_loss = sess.run(loss, feed_dict={x_data:rand_x, y_target:rand_y})

            loss_vec.append(temp_loss)
            train_acc_temp = sess.run(accuracy, feed_dict={x_data: x_val_train, y_target: y_val_train})
            train_accuracy.append(train_acc_temp)
            test_acc_temp = sess.run(accuracy, feed_dict={x_data: x_val_test, y_target: y_val_test})
            test_accuracy.append(test_acc_temp)

            print(temp_loss)


    with tf.name_scope('draw'):
        plt.plot(train_accuracy, 'r-')
        plt.plot(test_accuracy, 'b-')
        plt.show()

        plt.plot(loss_vec, 'r-')
        plt.show()
