import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sklearn.datasets as datasets

iris = datasets.load_iris()

def load_dataset(indices):
    return iris.data[indices], iris.target[indices]


def split():
    size = (iris.data.shape)[0]
    train_indices = np.random.choice(size, round(size * 0.8), replace=False)
    test_indices = np.array(list(set(range(size)) - set(train_indices)))
    return train_indices, test_indices


train_indices, test_indices = split()


def input(indices):
    data, target = load_dataset(indices)
    x_vals = np.array([[x[0], x[3]] for x in data])
    y_vals = np.array([1 if y == 0 else -1 for y in target])
    return np.matrix(x_vals), np.transpose(np.matrix(y_vals))


x_val_train, y_val_train = input(train_indices)
x_val_test, y_val_test = input(test_indices)

def get_data(x, y, batch_size=50):
    rand_index = np.random.choice(len(x), size=batch_size)
    rand_x = x[rand_index]
    rand_y = y[rand_index]
    return rand_x, rand_y


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
        for i in range(500):
            rand_x, rand_y = get_data(x_val_train, y_val_train, batch_size)
            sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

            temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
            loss_vec.append(temp_loss)

            train_acc_temp = sess.run(accuracy, feed_dict={x_data: x_val_train, y_target: y_val_train})
            train_accuracy.append(train_acc_temp)

            test_acc_temp = sess.run(accuracy, feed_dict={x_data: x_val_test, y_target: y_val_test})
            test_accuracy.append(test_acc_temp)

            if ((i+1) % 100) == 0:
                print("step# " + str(i+1) + " A = " + str(sess.run(A)) + " b = " + str(sess.run(b)))
                print("loss: " + str(temp_loss))
        [[a1], [a2]] = sess.run(A)
        b1 = sess.run(b)
        print(a1, a2, b1)


slope = -a2 / a1
y_intercept = (b1/a1)[0][0]
print(slope, y_intercept)

x_vals = np.array([[x[0], x[3]] for x in iris.data])
y_vals = np.array([1 if y == 0 else -1 for y in iris.target])

x1_vals = [d[1] for d in x_vals]
best_fits = []
for x in x1_vals:
    best_fits.append(x * slope + y_intercept)


setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == 1]
setosa_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == 1]

not_setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == -1]
not_setosa_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == -1]


plt.plot(setosa_x, setosa_y, 'o', label='I. setosa')
plt.plot(not_setosa_x, not_setosa_y, 'x', label='Non-setosa')
plt.plot(x1_vals, best_fits, 'r-', label='linear separator', linewidth=3)
plt.ylim([0, 10])
plt.legend(loc='lower right')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()


plt.plot(train_accuracy, 'k-', label='Training Accuracy')
plt.plot(test_accuracy, 'r--', label='Test Accuracy')
plt.title('Train and Test Set Accuracies')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()