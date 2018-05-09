import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data


class Model(object):
    """Abstracts a Tensorflow graph for a learning task.
    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs. Each algorithm you will construct in this homework will
    inherit from a Model object.
    """

    def __init__(self):
        with tf.name_scope("place_holder"):
            self._add_placeholders()

        self._train_op = self._add_model()

        self._summary = self._add_summary()

    def _add_placeholders(self):
        """Adds placeholder variables to tensorflow computational graph.
        Tensorflow uses placeholder variables to represent locations in a
        computational graph where data is inserted.  These placeholders are used as
        inputs by the rest of the model building code and will be fed data during
        training.
        See for more information:
        https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
        """
        self._x = tf.placeholder(tf.float32, [None, 784])
        self._label = tf.placeholder(tf.float32, [None, 10])
        self._keep_prob = tf.placeholder(tf.float32)

    def _add_model(self):
        """Implements core of model that transforms input_data into predictions.
        The core transformation for this model which transforms a batch of input
        data into a batch of predictions.
        Args:
          input_data: A tensor of shape (batch_size, n_features).
        Returns:
          out: A tensor of shape (batch_size, n_classes)
        """
        l1 = self._add_layer("layer1", self.x, 784, 150, activation_function=tf.sigmoid)

        l1_drop_out = tf.nn.dropout(l1, self.keep_prob)

        prediction = self._add_layer("layer2", l1_drop_out, 150, 10, activation_function=tf.nn.softmax)

        # add regular
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
        reg_term = tf.contrib.layers.apply_regularization(regularizer)

        self._loss = -tf.reduce_sum(self.label * tf.log(prediction)) + reg_term
        # 优化器选取 学习率设置 此处学习率置为0.1
        train_step = tf.train.AdamOptimizer(beta2=0.9999).minimize(self.loss)

        # for predict
        self.prediction = tf.argmax(prediction, 1)

        # for test
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.label, 1))
        self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        return train_step

    def run_epoch(self, sess, input_data, input_labels):
        """Runs an epoch of training.
        Trains the model for one-epoch.

        Args:
          sess: tf.Session() object
          input_data: np.ndarray of shape (n_samples, n_features)
          input_labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
          average_loss: scalar. Average minibatch loss of model on epoch.
        """

        return sess.run([self.loss, self.summary, self.train_op],
                        feed_dict={self.x: input_data, self.label: input_labels, self.keep_prob: 0.7})

    def fit(self, sess, mnist):
        """Fit model on provided data.
        Args:
          sess: tf.Session()
          input_data: np.ndarray of shape (n_samples, n_features)
          input_labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
          losses: list of loss per epoch
        """
        init = tf.global_variables_initializer()
        sess.run(init)
        writer = tf.summary.FileWriter("./logs", sess.graph)

        for i in range(5000):
            batch_xs, batch_ys = mnist.train.next_batch(50)
            loss, summary, _ = self.run_epoch(sess, batch_xs, batch_ys)
            writer.add_summary(summary, i)

    def _add_summary(self):
        with tf.name_scope("summary"):
            tf.summary.scalar("loss", self._loss)
            merged = tf.summary.merge_all()

        return merged

    def predict(self, sess, input_data, input_labels=None):
        """Make predictions from the provided model.
        Args:
          sess: tf.Session()
          input_data: np.ndarray of shape (n_samples, n_features)
          input_labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
          average_loss: Average loss of model.
          predictions: Predictions of model on input_data
        """
        return sess.run(self.correct_prediction,
                        feed_dict={self.x: input_data, self.label: input_labels, self.keep_prob: 1.0})

    def test(self, sess, input_data, input_labels):
        return sess.run(self.accuracy, feed_dict={self.x: input_data, self.label: input_labels, self.keep_prob: 1.0})

    @staticmethod
    def _add_layer(layerName, inputs, in_size, out_size, activation_function=None):
        # add one more layer and return the output of this layer
        with tf.variable_scope(layerName, reuse=None):
            weights = tf.get_variable("weights", shape=[in_size, out_size],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable("biases", shape=[1, out_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))

        tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights)

        Wx_plus_b = tf.matmul(inputs, weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

    @property
    def x(self):
        return self._x

    @property
    def label(self):
        return self._label

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def train_op(self):
        return self._train_op

    @property
    def correct_prediction(self):
        return self.prediction

    @property
    def loss(self):
        return self._loss

    @property
    def summary(self):
        return self._summary

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def keep_prob(self):
        return self._keep_prob


model = Model()
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

with tf.Session() as sess:
    model.fit(sess, mnist)
    print(model.test(sess, mnist.test.images, mnist.test.labels))
