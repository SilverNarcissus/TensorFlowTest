import tensorflow as tf
from sklearn import datasets
import numpy as np

def getTrainX():
    iris = datasets.load_iris()
    return np.append(iris.data[0:40], iris.data[50:90], axis=0)


def getTrainY():
    iris = datasets.load_iris()
    a = np.array([[1]])
    # loc = 1
    # for i in iris.target:
    #     b = np.array([int(i / 2), i % 2])
    #     a = np.row_stack((a, b))
    #     loc = loc + 1
    # a = np.delete(a, 0, 0)
    for i in np.append(iris.target[0:40], iris.target[50:90]):
        b = np.array([i])
        a = np.row_stack((a, b))
    return np.delete(a, 0, 0)


def getTestX():
    iris = datasets.load_iris()
    return np.append(iris.data[40:50], iris.data[90:100], axis=0)


def getTestY():
    iris = datasets.load_iris()
    a = np.array([[1]])
    for i in np.append(iris.target[40:50], iris.target[90:100]):
        b = np.array([i])
        a = np.row_stack((a, b))
    return np.delete(a, 0, 0)



cluster = tf.train.ClusterSpec({"ps": ["localhost:2222","localhost:2223"], "work": ["localhost:2224","localhost:2225","localhost:2226"]})
server = tf.train.Server(cluster, job_name="work", task_index=2)

# 添加隐藏层1
with tf.device("/job:ps/task:0"):
    xs = tf.placeholder(tf.float32, [None, 4])
    ys = tf.placeholder(tf.float32, [None, 1])
    Weights_1 = tf.get_variable("weights_1", shape=[4, 20],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases_1 = tf.get_variable("biases_1", shape=[1, 20],
                               initializer=tf.truncated_normal_initializer(stddev=0.1))

with tf.device("/job:worker/task:0"):
    Wx_plus_b_1 = tf.matmul(xs, Weights_1) + biases_1
    outputs_1 = tf.sigmoid(Wx_plus_b_1)
# 添加输出层
with tf.device("/job:ps/task:1"):
    Weights_2 = tf.get_variable("weights_2", shape=[20, 1],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases_2 = tf.get_variable("biases_2", shape=[1, 1],
                               initializer=tf.truncated_normal_initializer(stddev=0.1))

with tf.device("/job:worker/task:1"):
    Wx_plus_b_2 = tf.matmul(outputs_1, Weights_2) + biases_2
    outputs_2 = tf.sigmoid(Wx_plus_b_2)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - outputs_2), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session("grpc://localhost:2226") as sess:
    print("here")
    init = tf.global_variables_initializer()  # tensorflow更新后初始化所有变量不再用tf.initialize_all_variables()
    sess.run(init)
    for _ in range(200):
        sess.run(train_step, {xs: getTrainX(), ys: getTrainY()})
    print(sess.run(outputs_2, {xs: getTestX()}))