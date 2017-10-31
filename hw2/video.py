"""
Mask the entire dataset as a module.
"""
from os import listdir
from os.path import isfile, join
import json
import numpy as np

from pprint import pprint

import logging
logger = logging.getLogger()

class Video(object):
    def __init__(self, folder, dtype='train'):
        self._folder = folder
        self._dtype = Video._location_lookup(dtype)

        self._ids = self._list_ids()

    @staticmethod
    def _location_lookup(dtype):
        if dtype == 'train':
            return 'training_data'
        elif dtype == 'test':
            return 'testing_data'
        elif dtype == 'review':
                return 'peer_review'
        else:
            raise ValueError('Invalid type of dataset')

    def _list_ids(self):
        folder = join(self._folder, self._dtype, 'feat');
        file_list = [f for f in listdir(folder) if isfile(join(folder, f))]
        try:
            # remove the file extension
            self._ids = list(map(lambda s: s[:s.rindex('.avi.npy')], file_list))
            logger.info('{} IDs found'.format(len(self._ids)))
        except ValueError:
            logger.error('Invalid filename exists')

    def _load_data(self):
        self._load_labels()
        self._load_features()

    def _load_labels(self):
        file_path = '{}_label.json'.format(self._dtype)
        file_path = join(self._folder, file_path)
        with open(file_path, 'r') as fd:
            labels = json.load(fd)

        # format the input
        pprint(labels[0])

    def _load_features(self):
        folder = join(self._folder, self._dtype, 'feat');
        #file_list = glob(os.path.join(folder, '*.npy'))
        file_list = join(folder, 'xBePrplM4OA_6_18.avi.npy')

        data = np.load(file_list)
        pprint(data)
        print(data.shape)
