import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from Minist.NN_demo_class import NN


class CNN:
    def __init__(self):
        # 参数设置
        self.batch_size = 64
        self.input_x = 28
        self.input_y = 28
        self.output_size = 10
        # 多标签预测返回的个数
        self.multi_label_count = 3
        # 正则项系数
        self.regular_scale = 0.001

        # 建立模型相关量
        self.x, self.y, self.keep_prob = self.build_placeholder()
        self.row_prediction = self.build_model()
        # 下面是对于单标签分类的代码
        self.loss = self.build_one_lebal_loss()
        self.train_op = self.build_train_op()
        self.result = self.get_one_result()
        self.accuracy = self.one_result_accuracy()
        # 下面是对多标签分类的代码
        # self.loss = self.build_muti_lebal_loss()
        # self.train_op = self.build_train_op()
        # self.result_value, self.result_index = self.get_multi_result()

    # 返回训练时需要传入的 placeholder 的值
    def build_placeholder(self):
        # 声明一个占位符，None表示输入图片的数量不定，28*28图片分辨率
        x = tf.placeholder(tf.float32, [None, self.input_x * self.input_y])
        # 类别是0-9总共10个类别，对应输出分类结果
        y = tf.placeholder(tf.float32, [None, self.output_size])
        # dropout操作，减少过拟合，其实就是降低上一层某些输入的权重scale，甚至置为0，升高某些输入的权值，甚至置为2，防止评测曲线出现震荡，个人觉得样本较少时很必要
        # 使用占位符，由dropout自动确定scale，也可以自定义，比如0.5，根据tensorflow文档可知，程序中真实使用的值为1/0.5=2，也就是某些输入乘以2，同时某些输入乘以0
        keep_prob = tf.placeholder(tf.float32)

        return x, y, keep_prob

    # 建立这个模型
    # 返回值为之后的输出向量，shape为 batch_size * output_size
    def build_model(self):
        # x_image又把xs reshape成了28*28*1的形状，因为是灰色图片，所以通道是1.作为训练时的input，-1代表图片数量不定
        x_image = tf.reshape(self.x, [-1, self.input_x, self.input_y, 1])

        ## 第一层卷积操作 ##
        # 第一二参数值得卷积核尺寸大小，即patch，第三个参数是图像通道数，第四个参数是卷积核的数目，代表会出现多少个卷积特征图像;
        w_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=0.1),
                              collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
        # 对于每一个卷积核都有一个对应的偏置量。
        b_conv1 = tf.Variable(tf.truncated_normal(shape=[32], stddev=0.1))
        # 图片乘以卷积核，并加上偏执量，卷积结果28x28x32
        # 卷积遍历各方向步数为1，SAME：边缘外自动补0，遍历相乘
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
        # 池化结果14x14x32 卷积结果乘以池化卷积核
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        ## 第二层卷积操作 ##
        # 32通道卷积，卷积出64个特征
        w_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=0.1),
                              collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
        # 64个偏执数据
        b_conv2 = tf.Variable(tf.truncated_normal(shape=[64], stddev=0.1))
        # 注意h_pool1是上一层的池化结果，#卷积结果14x14x64
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
        # 池化结果7x7x64
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # 原图像尺寸28*28，第一轮图像缩小为14*14，共有32张，第二轮后图像缩小为7*7，共有64张

        ## 第三层全连接操作 ##
        # 二维张量，第一个参数7*7*64的patch，也可以认为是只有一行7*7*64个数据的卷积，第二个参数代表卷积个数共1024个
        W_fc1 = tf.Variable(tf.truncated_normal(shape=[7 * 7 * 64, 1024], stddev=0.1),
                            collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
        # 1024个偏执数据
        b_fc1 = tf.Variable(tf.truncated_normal(shape=[1024], stddev=0.1))
        # 将第二层卷积池化结果reshape成只有一行7*7*64个数据# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        # 卷积操作，结果是1*1*1024，单行乘以单列等于1*1矩阵，matmul实现最基本的矩阵相乘，不同于tf.nn.conv2d的遍历相乘，自动认为是前行向量后列向量
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # dropout操作，减少过拟合，其实就是降低上一层某些输入的权重scale，甚至置为0，升高某些输入的权值，甚至置为2，防止评测曲线出现震荡，个人觉得样本较少时很必要
        # 使用占位符，由dropout自动确定scale，也可以自定义，比如0.5，根据tensorflow文档可知，程序中真实使用的值为1/0.5=2，也就是某些输入乘以2，同时某些输入乘以0
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)  # 对卷积结果执行dropout操作

        # 第四层输出操作 ##
        # 二维张量，1*1024矩阵卷积，输出结果与 output_size 大小一致
        W_fc2 = tf.Variable(tf.truncated_normal(shape=[1024, self.output_size], stddev=0.1),
                            collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES])
        b_fc2 = tf.Variable(tf.truncated_normal(shape=[10], stddev=0.1))
        # 最后的分类，结果为batch_size * output_size
        prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        return prediction

    # 建立单结果预测的损失函数
    def build_one_lebal_loss(self):
        cross_entropy = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.row_prediction))
        reg_term = self._build_regular_term()
        loss = cross_entropy + reg_term

        return loss

    # 建立正则项
    def _build_regular_term(self):
        regularizer = tf.contrib.layers.l2_regularizer(scale=self.regular_scale)
        reg_term = tf.contrib.layers.apply_regularization(regularizer)
        return reg_term

    # 建立多结果预测的损失函数
    def build_muti_lebal_loss(self):
        cross_entropy = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.row_prediction))
        reg_term = self._build_regular_term()
        loss = cross_entropy + reg_term

        return loss

    # 建立训练张量
    def build_train_op(self):
        train_op = tf.train.AdamOptimizer(beta2=0.9999).minimize(self.loss)

        return train_op

    # 拟合数据集
    def get_one_result(self):
        soft_max = tf.nn.softmax(self.row_prediction)
        result = tf.argmax(soft_max, 1)

        return result

    # 得到单标签准确度
    def one_result_accuracy(self):
        soft_max = tf.nn.softmax(self.row_prediction)
        correct_prediction = tf.equal(tf.argmax(soft_max, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return accuracy

    def get_multi_result(self):
        soft_max = tf.nn.softmax(self.row_prediction)
        # value -> 对应的是 top k 的概率值
        # index -> 对应的是 top k 的下标
        # 举个例子 [1,5,2,4,6]  top 2 : value -> [6, 5]  index -> [4,1]
        value, index = tf.nn.top_k(soft_max, k=self.multi_label_count)

        return value, index


#########################
# 下面是训练代码
model = NN()
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)  # 读取图片数据集
iteration = 10

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 保存参数所用的保存器
    saver = tf.train.Saver(max_to_keep=1)
    # get latest file
    ckpt = tf.train.get_checkpoint_state('/Users/SilverNarcissus/PycharmProjects/TensorFlowTest/modelStore')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    # 可视化部分
    tf.summary.scalar("loss", model.loss)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs", sess.graph)

    for i in range(iteration):
        batch = mnist.train.next_batch(model.batch_size)
        if i % 100 == 0:
            train_accuracy = sess.run(model.accuracy,
                                      feed_dict={model.x: batch[0], model.y: batch[1], model.keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))

        if i > 0 and i % 1000 == 0:
            saver.save(sess, "/Users/SilverNarcissus/PycharmProjects/TensorFlowTest/modelStore/CNN", global_step=i)

        _, summary = sess.run([model.train_op, merged],
                              feed_dict={model.x: batch[0], model.y: batch[1], model.keep_prob: 0.7})
        writer.add_summary(summary, i)

    print("test accuracy %g" % sess.run(model.accuracy,
                                        feed_dict={model.x: mnist.test.images, model.y: mnist.test.labels,
                                                   model.keep_prob: 1.0}))
