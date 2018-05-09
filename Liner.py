import tensorflow as tf
import random

x = tf.placeholder(tf.float32)
w = tf.Variable(-1.0)
b = tf.Variable(0.0)
y = tf.multiply(x, w)
y = tf.add(y, b)
y_true = tf.placeholder(tf.float32)

loss = tf.pow((y - y_true), 2)
optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)

x_ = [i for i in range(100)]
y_ = [i + 4.5 + random.random() for i in range(100)]

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)

    for n in range(10):
        for i in range(100):
            print(x_[i], y_[i])
            _, w_, b_, y1 = sess.run([optimizer, w, b, loss], feed_dict={x: x_[i], y_true: y_[i]})
            print(w_, b_, y1)
