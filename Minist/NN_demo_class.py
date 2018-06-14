import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class NN:
    def __init__(self):
        # 参数设置
        self.training_batch_size = 256
        self.valid_batch_size = 256
        self.iteration = 100000
        # embediing size = 128
        self.input_size = 28 * 28
        # accu_size = 203
        self.output_size = 10
        # 多标签预测返回的个数
        self.multi_label_count = 3

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

    # 增加一层神经网络的抽象函数
    def _add_layer(self, layerName, inputs, in_size, out_size, activation_function=None):
        # add one more layer and return the output of this layer
        with tf.variable_scope(layerName, reuse=None):
            Weights = tf.get_variable("weights", shape=[in_size, out_size],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable("biases", shape=[1, out_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))

        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, Weights)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

    # 返回训练时需要传入的 placeholder 的值
    def build_placeholder(self):
        x = tf.placeholder(tf.float32, [None, self.input_size])
        y = tf.placeholder(tf.float32, [None, self.output_size])
        # dropout操作，减少过拟合，其实就是降低上一层某些输入的权重scale，甚至置为0，升高某些输入的权值，甚至置为2，防止评测曲线出现震荡，个人觉得样本较少时很必要
        # 使用占位符，由dropout自动确定scale，也可以自定义，比如0.5，根据tensorflow文档可知，程序中真实使用的值为1/0.5=2，也就是某些输入乘以2，同时某些输入乘以0
        keep_prob = tf.placeholder(tf.float32)

        return x, y, keep_prob

    # 建立这个模型
    # 返回值为之后的输出向量，shape为 batch_size * output_size
    def build_model(self):
        # 添加隐藏层1
        l1 = self._add_layer("layer1", self.x, self.input_size, 64, activation_function=tf.sigmoid)
        # 添加隐藏层2
        # l2 = add_layer("layer2", l1, 256, 256, activation_function=tf.sigmoid)
        l1_drop = tf.nn.dropout(l1, self.keep_prob)
        # 添加输出层
        prediction = self._add_layer("layer3", l1_drop, 64, self.output_size, activation_function=tf.identity)

        return prediction

    # 建立单结果预测的损失函数
    def build_one_lebal_loss(self):
        cross_entropy = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.row_prediction))
        reg_term = self._build_regular_term()
        loss = cross_entropy + reg_term

        return loss

    def _build_regular_term(self):
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
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
iteration = 1000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 保存参数所用的保存器
    saver = tf.train.Saver(max_to_keep=1)
    # get latest file
    ckpt = tf.train.get_checkpoint_state('/Users/SilverNarcissus/PycharmProjects/TensorFlowTest/modelStore/nn')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    # 可视化部分
    tf.summary.scalar("loss", model.loss)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs", sess.graph)

    for i in range(iteration):
        batch = mnist.train.next_batch(model.training_batch_size)
        if i % 100 == 0:
            train_accuracy = sess.run(model.accuracy,
                                      feed_dict={model.x: batch[0], model.y: batch[1], model.keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))

        if i > 0 and i % 1000 == 0:
            saver.save(sess, "/Users/SilverNarcissus/PycharmProjects/TensorFlowTest/modelStore/nn/NN", global_step=i)

        _, summary = sess.run([model.train_op, merged],
                              feed_dict={model.x: batch[0], model.y: batch[1], model.keep_prob: 0.7})
        writer.add_summary(summary, i)

    print("test accuracy %g" % sess.run(model.accuracy,
                                        feed_dict={model.x: mnist.test.images, model.y: mnist.test.labels,
                                                   model.keep_prob: 1.0}))
