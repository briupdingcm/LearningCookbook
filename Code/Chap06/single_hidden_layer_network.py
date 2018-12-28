import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sklearn.datasets as datasets


seed = 3
tf.set_random_seed(seed)
np.random.seed(seed)


input_layer_nodes = 3
hidden_layer_nodes = 10
out_layer_nodes = 1


iteration = 500
batch_size = 50
learning_rate = 0.005
g = tf.Graph()

iris = datasets.load_iris()

def load():
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target
    return zip(data, target)


# Normalize by column (min-max norm)
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)



def split():
    dataset = load()
    d = np.array([[x, y] for (x, y) in dataset])
    num = len(d)
    train_indices = np.random.choice(num, round(num * 0.8), replace=False)
    test_indices = np.array(list(set(range(num)) - set(train_indices)))
    train_data = d[train_indices]
    test_data = d[test_indices]
    return train_data, test_data

train_data, test_data = split()


def input(dataset):
    x_vals = np.nan_to_num(normalize_cols(np.array([(x[0])[0:3] for x in dataset])))
    y_vals = np.array([[(x[0])[3]] for x in dataset])
    return x_vals, y_vals

x_vals_train, y_vals_train = input(train_data)
x_vals_test, y_vals_test = input(test_data)

def get(x, y):
    rand_index = np.random.choice(len(x), size=batch_size)
    return np.array(x[rand_index]), np.array(y[rand_index])



with g.as_default():
    x_data = tf.placeholder(shape=[None, input_layer_nodes], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, out_layer_nodes], dtype=tf.float32)

    W1 = tf.Variable(tf.random_normal(shape=[input_layer_nodes, hidden_layer_nodes], dtype=tf.float32))
    b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))  # one biases for each hidden node

    W2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, out_layer_nodes], dtype=tf.float32))
    b2 = tf.Variable(tf.random_normal(shape=[out_layer_nodes], dtype=tf.float32))  # 1 bias for the output

    with tf.name_scope('inference'):
        hidden_output = tf.nn.relu(tf.add(tf.matmul(x_data, W1), b1))
        final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output, W2), b2))

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(y_target - final_output))

    with tf.name_scope('train'):
        my_opt = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = my_opt.minimize(loss)

    loss_vec = []
    test_loss = []

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(iteration):
            rand_x, rand_y = get(x_vals_train, y_vals_train)
            sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
            temp_loss = sess.run(loss, feed_dict={x_data: x_vals_train, y_target: y_vals_train})
            loss_vec.append(temp_loss)

            test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: y_vals_test})
            test_loss.append(test_temp_loss)
            if (i + 1) % 50 == 0:
                print('Generation: ' + str(i + 1) + '. Loss = ' + str(temp_loss))

    with tf.name_scope('draw'):
        # Plot loss (MSE) over time
        plt.plot(loss_vec, 'k-', label='Train Loss')
        plt.plot(test_loss, 'r--', label='Test Loss')
        plt.title('Loss (MSE) per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.show()