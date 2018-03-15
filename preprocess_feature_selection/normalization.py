import tensorflow as tf


def normalize(input_data):
    mean, var = tf.nn.moments(input_data, 0)
    input_data = (input_data - mean) / tf.sqrt(var)
    return input_data
