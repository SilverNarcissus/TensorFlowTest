import tensorflow as tf
from irisData import getTrainX
import numpy as np


class PCA:
    def __init__(self, fit_data, session, keep_k=None, keep_ratio=None):
        self._keep_k = keep_k
        self._keep_ratio = keep_ratio
        self._fit(fit_data, session)

    def _fit(self, fit_data, sess):
        m, n = fit_data.get_shape()
        sigma = tf.zeros(shape=[n, n])
        for i in range(m):
            vector = tf.strided_slice(fit_data, [i, 0], [i + 1, n])
            sigma += tf.matmul(tf.transpose(vector), vector)

        sigma = sigma / tf.constant(m.value, dtype=tf.float32)
        s, u, _ = tf.svd(sigma)
        if self.keep_k is None:
            self._get_keep_k(sess.run(s))
            print(self.keep_k)

        z = tf.strided_slice(u, [0, 0], [n, self.keep_k])
        self._z = sess.run(z)

    def _get_keep_k(self, s):
        print(s)
        s = s / np.sum(s)
        total = 0
        for i in range(len(s)):
            total += s[i]
            if total >= self._keep_ratio:
                self._keep_k = i + 1
                return

    def transfer(self, input_data):
        return tf.matmul(input_data, self.z)

    @property
    def keep_k(self):
        return self._keep_k

    @property
    def keep_ratio(self):
        return self._keep_ratio

    @property
    def z(self):
        return self._z


x = tf.constant([[1, 1, 1], [2, 2, 10]], dtype=tf.float32)
with tf.Session() as sess:
    input_data = tf.constant(getTrainX(), dtype=tf.float32)
    p = PCA(input_data, sess, keep_ratio=0.95)
    print(sess.run(p.transfer(input_data)))
