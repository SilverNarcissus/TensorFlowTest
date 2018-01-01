import tensorflow as tf

from irisData import getTrainY, getTrainX, getTestX


def add_layer(layerName, inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    with tf.variable_scope(layerName, reuse=None):
        Weights = tf.get_variable("weights", shape=[in_size, out_size],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", shape=[1, out_size],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))

    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 4])
ys = tf.placeholder(tf.float32, [None, 1])
# 添加隐藏层1
l1 = add_layer("layer1", xs, 4, 20, activation_function=tf.sigmoid)
# 添加隐藏层2
l2 = add_layer("layer2", l1, 20, 200, activation_function=tf.sigmoid)
# 添加输出层
prediction = add_layer("layer3", l2, 200, 1, activation_function=tf.sigmoid)
# MSE 均方误差
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
# 优化器选取 学习率设置 此处学习率置为0.1
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# tensorflow变量初始化，打开会话
init = tf.global_variables_initializer()  # tensorflow更新后初始化所有变量不再用tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train_step, {xs: getTrainX(), ys: getTrainY()})

print(sess.run(prediction, {xs : getTestX()}))
# count = 0
# for out in sess.run(prediction, {xs : getTestX()}):
#     if(out[0] > 0.5):
#         print(str(count) + ':1')
#     else:
#         print(str(count) + ':0')
#     count = count + 1
