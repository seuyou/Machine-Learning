import tensorflow as tf
import os
import time
epoch = 4000

# author: Yuan Liu
# date: 2019/4/21

time_stamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "run", time_stamp))

dataset = tf.data.Dataset.from_tensor_slices(([[[0., 0.]], [[1., 0.]], [[0., 1.]], [[1., 1.]]],
                                              [[[0., 1.]], [[0., 1.]], [[0., 1.]], [[1., 0.]]]))
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
x_placeholder = tf.placeholder(tf.float32, [None, 2], name="x")
y_placeholder = tf.placeholder(tf.float32, [None, 2], name="y")
global_step = tf.get_variable("global_step", [], dtype=tf.int64, initializer=tf.constant_initializer(0),
                              trainable=False)
W = tf.get_variable("W", shape=[2, 2], dtype=tf.float32,
                    initializer=tf.random_uniform_initializer(minval=-5, maxval=5))
b = tf.get_variable("b", shape=[2], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
scores = tf.nn.xw_plus_b(x_placeholder, W, b)
losses = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_placeholder, logits=scores))
correct_prediction = tf.equal(tf.argmax(scores, 1), tf.argmax(y_placeholder, 1))
prediction = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
optimizer = tf.train.AdamOptimizer(1e-3).minimize(losses, global_step=global_step)

loss_scalar = tf.summary.scalar("loss", losses)
accuracy_scalar = tf.summary.scalar("accuracy", prediction)

with tf.Session() as sess:
    train_dir = os.path.join(out_dir, "summary", "train")
    train_summary_op = tf.summary.merge([loss_scalar, accuracy_scalar])
    train_summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

    checkpoint_dir = os.path.join(out_dir, "checkpoint")
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(epoch):
        sess.run(iterator.initializer)
        while True:
            try:
                x, y = sess.run(next_element)
                _, loss, accuracy, summaries, step = sess.run([optimizer, losses, prediction, train_summary_op,
                                                               global_step],
                                                              feed_dict={x_placeholder: x,
                                                                         y_placeholder: y})
                train_summary_writer.add_summary(summaries, step)
                if step % 50 == 0:
                    path = saver.save(sess, checkpoint_prefix)
                    print("saving to {}".format(path))
                print("loss:{}, accuracy:{}".format(loss, accuracy))
            except tf.errors.OutOfRangeError:
                break
