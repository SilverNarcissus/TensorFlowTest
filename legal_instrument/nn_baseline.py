# This model is a base line model using neural network

import legal_instrument.generate_batch as generator
import tensorflow as tf
import legal_instrument.system_path as constant

# param
training_batch_size = 128
valid_batch_size = 1024
embedding_size = 128
iteration = 10000
##

accu_dict, reverse_accu_dict = generator.read_accu()
word_dict, embedding, reverse_dictionary = generator.get_dictionary_and_embedding()

print("reading data from training set...")
train_data_x, train_data_y = generator.read_data(constant.DATA_TRAIN, len(accu_dict), embedding, word_dict, accu_dict)
valid_data_x, valid_data_y = generator.read_data(constant.DATA_VALID, len(accu_dict), embedding, word_dict, accu_dict)
print("reading complete!")

# just test generate_accu_batch
x, y = generator.generate_accu_batch(training_batch_size, train_data_x, train_data_y, len(accu_dict))

print("data load complete")
print("The model begin here")


# 增加一层神经网络的抽象函数
def add_layer(layerName, inputs, in_size, out_size, activation_function=None):
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


xs = tf.placeholder(tf.float32, [None, embedding_size])
ys = tf.placeholder(tf.float32, [None, len(accu_dict)])
# 添加隐藏层1
l1 = add_layer("layer1", xs, embedding_size, 200, activation_function=tf.sigmoid)
# 添加隐藏层2
# l2 = add_layer("layer2", l1, 20, 15, activation_function=tf.sigmoid)
# 添加输出层
prediction = add_layer("layer3", l1, 200, len(accu_dict), activation_function=tf.identity)

# 添加正则项
regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
reg_term = tf.contrib.layers.apply_regularization(regularizer)
# 损失函数
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction)) + reg_term

# 优化器选取
train_step = tf.train.AdamOptimizer().minimize(loss)

# 评价部分
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# run part
with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    # 保存参数所用的保存器
    saver = tf.train.Saver(max_to_keep=2)
    # get latest file
    ckpt = tf.train.get_checkpoint_state('./modelStore')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    # 可视化部分
    tf.summary.scalar("loss", loss)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs", sess.graph)

    # training part
    for i in range(iteration):
        x, y = generator.generate_accu_batch(training_batch_size, train_data_x, train_data_y, len(accu_dict))

        _, summary = sess.run([train_step, merged], feed_dict={xs: x, ys: y})
        writer.add_summary(summary, i)

        if i % 1000 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={xs: x, ys: y})
            valid_x, valid_y = generator.generate_accu_batch(valid_batch_size, valid_data_x, valid_data_y,
                                                             len(accu_dict))
            valid_accuracy = sess.run(accuracy, feed_dict={xs: valid_x, ys: valid_y})
            print("step %d, training accuracy %g" % (i, train_accuracy))
            print("step %d, valid accuracy %g" % (i, valid_accuracy))

            saver.save(sess, "./modelStore/base_line", global_step=i)
