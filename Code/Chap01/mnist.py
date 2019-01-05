import my_data as md

mnist = md.read_data_sets("/Users/kevinding/MNIST/", one_hot=True)
print(len(mnist.train.images))
print(len(mnist.test.images))
print(len(mnist.validation.images))
print(mnist.train.labels[1, :])

