import os
import logging
logger = logging.getLogger(__name__)

import tensorflow as tf
import tensorflow.contrib as tc

import numpy as np
import matplotlib.pyplot as plt

import skimage.io

def split(x):
    assert type(x) == int
    t = int(np.floor(np.sqrt(x)))
    for a in range(t, 0, -1):
        if x % a == 0:
            return a, int(x / a)

def grid_transform(x, size):
    a, b = split(x.shape[0])
    h, w, c = size[0], size[1], size[2]
    x = np.reshape(x, [a, b, h, w, c])
    x = np.transpose(x, [0, 2, 1, 3, 4])
    x = np.reshape(x, [a * h, b * w, c])
    if x.shape[2] == 1:
        x = np.squeeze(x, axis=2)
    return x

def grid_show(fig, x, size):
    ax = fig.add_subplot(111)
    x = grid_transform(x, size)
    if len(x.shape) > 2:
        ax.imshow(x)
    else:
        ax.imshow(x, cmap='gray')

class WassersteinGAN(object):
    def __init__(self, g_net, d_net, data_sampler, noise_sampler, scale=10.0):
        '''
        Parameters
        ----------
        g_net : model
            The generator.
        d_net : model
            The discriminator.
        data_sampler: data
            The input data.
        noise_sampler: data
            The noise generator.
        '''
        self.g_net = g_net
        self.d_net = d_net
        self.data_sampler = data_sampler
        self.noise_sampler = noise_sampler
        self.image_dim = self.d_net.image_dim
        self.label_dim = self.g_net.label_dim
        self.noise_dim = self.g_net.noise_dim
        self.images = tf.placeholder(tf.float32, [None, self.image_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.label_dim], name='labels')
        self.noise = tf.placeholder(tf.float32, [None, self.noise_dim], name='noise')

        self._images = self.g_net(self.noise, self.labels)

        # discriminate real images
        self.d = self.d_net(self.images, self.labels, reuse=False)
        # discriminate fake images
        self._d = self.d_net(self._images, self.labels)

        d_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(tf.nn.sigmoid(self.d)), logits=self.d
            )
        )
        _d_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(tf.nn.sigmoid(self._d)), logits=self._d
            )
        )

        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(tf.nn.sigmoid(self._d)), logits=self._d
            )
        )

        self.g_loss = g_loss
        self.d_loss = d_loss + _d_loss

        epsilon = tf.random_uniform([], 0.0, 1.0)
        image_hat = epsilon*self.images + (1-epsilon)*self._images
        d_hat = self.d_net(image_hat, self.labels)

        ddx = tf.gradients(d_hat, image_hat)[0]
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx-1.0) * scale)
        self.d_loss += ddx

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.g_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)\
                .minimize(self.g_loss, var_list=self.g_net.vars)
            self.d_adam = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)\
                .minimize(self.d_loss, var_list=self.d_net.vars)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()

    def train(self, epochs=100, batch_size=64, d_iters=5):
        self.sess.run(tf.global_variables_initializer())
        for t in range(epochs):
            bx, by = self.data_sampler(batch_size)

            for _ in range(d_iters):
                bz = self.noise_sampler(batch_size, self.noise_dim)
                self.sess.run(
                    self.d_adam, feed_dict={
                        self.images: bx,
                        self.labels: by,
                        self.noise: bz
                    }
                )

            bz = self.noise_sampler(batch_size, self.noise_dim)
            self.sess.run(
                self.g_adam, feed_dict={
                    self.images: bx,
                    self.labels: by,
                    self.noise: bz
                }
            )

            if t % 100 == 0:
                bx, by = self.data_sampler(batch_size)
                bz = self.noise_sampler(batch_size, self.noise_dim)

                g_loss = self.sess.run(
                    self.g_loss, feed_dict={
                        self.images: bx,
                        self.labels: by,
                        self.noise: bz
                    }
                )
                d_loss = self.sess.run(
                    self.d_loss, feed_dict={
                        self.images: bx,
                        self.labels: by,
                        self.noise: bz
                    }
                )
                logger.info('Iter {}, g_loss {:.4f}, d_loss {:.4f}'\
                    .format(t, g_loss, d_loss))

                # save model
                save_path = self.saver.save(self.sess, 'saved_model/model.ckpt')
                logger.info('checkpoint saved to \'{}\' at t={}'\
                    .format(save_path, t))

                bx = self.sess.run(
                    self._images,
                    feed_dict={
                        self.labels: by,
                        self.noise: bz
                    }
                )
                # save images
                save_path = os.path.join('logs', '{}_g{:.4f}_d{:.4f}'\
                    .format(t, g_loss, d_loss))
                os.makedirs(save_path)
                bx = self.data_sampler.to_images(bx)
                for i in range(bx.shape[0]):
                    skimage.io.imsave(os.path.join(
                        save_path, '{}.jpg'.format(i)), bx[i, ...]
                    )

            """
            if t % 100 == 0:
                bz = self.noise_sampler(batch_size, self.noise_dim)
                bx = self.sess.run(
                    self._images, feed_dict={self.noise: bz, self.labels: }
                )

                # save images
                bx = self.data_sampler.to_images(bx)
                for i in range(bx.shape[0]):
                    skimage.io.imsave('logs/{}_{}.jpg'.format(t/100, i), bx[i, ...])

                # save preview
                fig = plt.figure('WGAN')
                grid_show(fig, bx, self.data_sampler.shape)
                fig.savefig('logs/{}.pdf'.format(t/100))

                # save model variables
                save_path = self.saver.save(self.sess, 'saved_model/model.ckpt')
            """
