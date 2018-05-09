import numpy as np
import tensorflow as tf
import random

xs = tf.placeholder(tf.float32, [None, 2])
ys = tf.placeholder(tf.float32, [None, 1])

weights = tf.get_variable("weights", shape=[2, 1],
                          initializer=tf.truncated_normal_initializer(stddev=5))

biases = tf.get_variable("biases", shape=[1, 1],
                         initializer=tf.truncated_normal_initializer(stddev=5))

predict = tf.matmul(xs, weights) + biases
result = tf.nn.sigmoid(predict)

loss = -(ys * tf.log(result))
loss = tf.reduce_sum(loss)
train_step = tf.train.AdamOptimizer().minimize(loss)

init = tf.global_variables_initializer()

hashtable = {"五食堂": 1, "篮球场": 1, "操场": -1, "体育馆": 1, "四": 1, "七食堂": -1, "八食堂": -1,
             "四食堂": 1, "六食堂": 1, "图书馆": -1, "校门": -1, "马路": -1}

words = ["五食堂,边上", "体育馆,边上", "四食堂,边上", "篮球场,边上", "六食堂,边上", "五食堂,附近",
         "六食堂,附近", "六食堂,附近", "体育馆,附近", "靠近,四食堂", "靠近,五食堂", "靠近,六食堂", "靠近,体育馆",
         "操场,附近", "校门,附近", "操场,边上", "图书馆,边上", "图书馆,附近", "马路,附近", "靠近,图书馆", "靠近,操场",
         "靠近,马路", "临近,马路", "临近,图书馆", "临近,操场"]
label = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def random_pick(num):
    result = []
    y = []
    for i in range(num):
        loc = random.randint(0, 24)
        x = []
        for word in words[loc].split(","):
            if word in hashtable.keys():
                x.append(hashtable[word])
            else:
                x.append(0)
        j = len(x)
        while j < 2:
            x.append(0)
            j = j + 1

        y.append(label[loc])
        result.append(x)

    result.append(y)
    return result


def f(sen):
    x = []
    for word in sen.split(","):
        if word in hashtable.keys():
            x.append(hashtable[word])
        else:
            x.append(0)
    j = len(x)
    while j < 2:
        x.append(0)
        j = j + 1

    result = []
    result.append(x)
    return result


with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        data = random_pick(5)
        y = data[5:6]
        y_ = np.reshape(y, [5, 1])
        x_ = np.reshape(data[0:5], [5, 2])
        sess.run(train_step, feed_dict={xs: x_, ys: y_})
        if i % 50 == 0:
            print(sess.run(loss, feed_dict={xs: x_, ys: y_}))

    print(sess.run(result, feed_dict={xs: f("操场,入口"), ys: [[-10]]}))
