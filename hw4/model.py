import tensorflow as tf
import tensorflow.contrib.layers as tcl

def leaky_relu_batch_norm(x, alpha=0.2):
    return tf.nn.leaky_relu(tcl.batch_norm(x), alpha)

def relu_batch_norm(x):
    return tf.nn.relu(tcl.batch_norm(x))

class Generator(object):
    def __init__(self):
        self.noise_dim = 100
        self.label_dim = 23
        self.name = 'generator'

    def __call__(self, noise, labels):
        with tf.variable_scope(self.name) as vs:
            bs = tf.shape(noise)[0]
            noise = tf.concat([noise, labels], 1)
            fc = tcl.fully_connected(noise, 4 * 4 * 1024, activation_fn=tf.identity)
            conv1 = tf.reshape(fc, tf.stack([bs, 4, 4, 1024]))
            conv1 = relu_batch_norm(conv1)
            conv2 = tcl.conv2d_transpose(
                conv1, 512, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=relu_batch_norm
            )
            conv3 = tcl.conv2d_transpose(
                conv2, 256, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=relu_batch_norm
            )
            conv4 = tcl.conv2d_transpose(
                conv3, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=relu_batch_norm
            )
            conv5 = tcl.conv2d_transpose(
                conv4, 3, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.tanh
            )
            return conv5

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class Discriminator(object):
    def __init__(self):
        self.image_dim = 64 * 64 * 3
        self.name = 'discriminator'

    def __call__(self, images, labels, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(images)[0]
            x = tf.reshape(images, [bs, 64, 64, 3])
            conv1 = tcl.conv2d(
                images, 64, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.nn.leaky_relu
            )
            conv2 = tcl.conv2d(
                conv1, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu_batch_norm
            )
            conv3 = tcl.conv2d(
                conv2, 256, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu_batch_norm
            )
            conv4 = tcl.conv2d(
                conv3, 512, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=leaky_relu_batch_norm
            )
            #TODO append broadcasted label dimension at the end of conv4 (last dimension)
            conv4 = tcl.flatten(conv4)
            fc = tcl.fully_connected(conv4, 1, activation_fn=tf.identity)
            return fc

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
