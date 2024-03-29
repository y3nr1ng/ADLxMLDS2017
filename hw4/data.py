import os
import logging
logger = logging.getLogger(__name__)

import skimage.io
import skimage.transform
import numpy as np
import pandas as pd

import utils

class DataSampler(object):
    def __init__(self, path):
        self.shape = [64, 64, 3]
        if path is None:
            return
        self.db_path = os.path.join('data', 'faces')
        self.db_files, self.labels = self.list_valid_files(path)
        self.cur_batch_ptr = 0
        self.cur_batch_data, markers = self.load_new_data()
        self.cur_batch_label = self.load_label_range(markers)
        self.train_batch_ptr = 0

    def list_valid_files(self, tag_path):
        df = pd.read_csv(tag_path, index_col=0, names=['id', 'tags'])
        db_files = ['{}.jpg'.format(i) for i in df.index.values.tolist()]
        return db_files, utils.text_to_onehot(df)

    def load_new_data(self, batch_size=None):
        if not batch_size:
            batch_size = len(self.db_files)

        if self.cur_batch_ptr >= len(self.db_files):
            self.cur_batch_ptr = 0

        i_start = self.cur_batch_ptr
        x = [self.load_single_image() for _ in range(batch_size)]
        i_end = self.cur_batch_ptr

        return np.stack(x, axis=0), (i_start, i_end)

    def load_label_range(self, markers):
        i_start, i_end = markers
        if i_end < i_start:
            y1 = self.labels[i_start:, :]
            y2 = self.labels[:i_end, :]
            y = np.stack([y1, y2], axis=0)
        else:
            y = self.labels[i_start:i_end, :]
        return y

    def load_single_image(self):
        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                self.db_path, self.db_files[self.cur_batch_ptr])
        self.cur_batch_ptr += 1
        x = skimage.io.imread(filename)
        x = skimage.transform.resize(x, self.shape[:2], mode='constant')
        return x * 2.0 - 1.0

    def __call__(self, batch_size):
        prev_batch_ptr = self.train_batch_ptr
        self.train_batch_ptr += batch_size
        if self.train_batch_ptr > self.cur_batch_ptr:
            self.train_batch_ptr = batch_size
            prev_batch_ptr = 0
            self.cur_batch_data, markers = self.load_new_data()
            self.cur_batch_label = self.load_label_range(markers)
        x = self.cur_batch_data[prev_batch_ptr:self.train_batch_ptr, :, :, :]
        y = self.cur_batch_label[prev_batch_ptr:self.train_batch_ptr, :]
        return np.reshape(x, [batch_size, -1]), np.reshape(y, [batch_size, -1])

    def to_images(self, data):
        rescaled = np.divide(data + 1.0, 2.0)
        return np.reshape(np.clip(rescaled, 0.0, 1.0), [data.shape[0]] + self.shape)

class NoiseSampler(object):
    def __init__(self, seed=None):
        np.random.seed(seed)

    def __call__(self, batch_size, noise_dim):
        return np.random.uniform(-1.0, 1.0, [batch_size, noise_dim])
