import tensorflow as tf
import numpy as np


class gate_graph():

    def __init__(self):

        self.input_x = tf.placeholder(tf.float32, [1, 2], "x")
        self.w = tf.get_variable("W", [2, 2], dtype=tf.float32, initializer=tf.random_uniform_initializer)
        self.b = tf.get_variable("b", [2], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        self.prediction = tf.sigmoid(tf.nn.xw_plus_b(self.input_x, self.w, self.b))


def gate(in1, in2):
    tf.reset_default_graph()
    input_a = np.array([in1, in2]).astype(float)
    input_a = np.reshape(input_a, [1, 2])
    gate_g = gate_graph()
    saver = tf.train.Saver({"W": gate_g.w, "b": gate_g.b})
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        add = r"C:\Users\liuyuan\Desktop\ML\logistic_regression\run\1555826495\checkpoint\model"
        saver.restore(sess, add)
        pred = sess.run(gate_g.prediction, feed_dict={gate_g.input_x: input_a})
        if np.argmax(pred, -1) == 0:
            return 1
        else:
            return 0



