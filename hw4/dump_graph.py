import os

import tensorflow as tf

saver = tf.train.import_meta_graph(
    os.path.join('saved_model', 'model.ckpt.meta')
)
with tf.Session() as sess:
    saver.restore(sess, os.path.join('saved_model', 'model.ckpt'))
    writer = tf.summary.FileWriter('logs', sess.graph)
