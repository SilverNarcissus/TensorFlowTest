import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.train.images[0])
print(mnist.train.labels[0])

# weight = tf.get_variable("weight", shape=[784, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
# b = tf.get_variable("bias", shape=[1, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
# x = tf.placeholder(tf.float32, [None, 784])
# evidence = tf.matmul(x, weight) + b
# model = tf.nn.softmax(evidence)
#
# y_ = tf.placeholder("float", [None, 10])
# cross_entropy = -tf.reduce_sum(y_ * tf.log(model))
# train_step = tf.train.AdamOptimizer(beta2=0.9999).minimize(cross_entropy)
#
# result = tf.argmax(model, 1)
#
# init = tf.initialize_all_variables()
#
# sess = tf.Session()
# sess.run(init)
#
# for i in range(1000):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
#
# correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#
# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#
# batch_xs, batch_ys = mnist.test.next_batch(5)
# print(sess.run(result, feed_dict={x: batch_xs, y_: batch_ys}))
# print(batch_ys)

