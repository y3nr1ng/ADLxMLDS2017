"""
Mask the entire dataset as a module.
"""
from os import listdir
from os.path import isfile, join, exists
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
        # load all the data, features and labels
        self._data = self._load_data()

        # build word dictionary
        self._dict = self._build_dict()

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
        # generate id file name
        prefix = self._dtype
        prefix = prefix[:-5] if self._dtype.endswith('_data') else prefix
        id_file = '{}_id.txt'.format(prefix)
        # load ids from file if exists
        if isfile(id_file):
            with open(id_file, 'r') as fd:
                self._ids = [line.rstrip('\n') for line in fd]
        else:
            folder = join(self._folder, self._dtype, 'feat');
            file_list = [f for f in listdir(folder) if isfile(join(folder, f))]
            try:
                # remove the file extension
                self._ids = list(map(lambda s: s[:s.rindex('.npy')], file_list))
            except ValueError:
                logger.error('Invalid filename exists')
        logger.info('{} IDs found'.format(len(self._ids)))

    def _load_data(self):
        # create empty dataset, nested dict
        data = dict.fromkeys(self._ids, {})

        data = self._load_labels(data)
        data = self._load_features(data)

        raise RuntimeError("_load_data")

    def _load_labels(self, data):
        """
        Load the dictionary representation of labels.
        """
        # generate label file name
        prefix = self._dtype
        prefix = prefix[:-5] if self._dtype.endswith('_data') else prefix
        label_file = '{}_label.json'.format(prefix)
        # load ids from file if exists
        if isfile(label_file):
            with open(label_file, 'r') as fd:
                labels = json.load(fd)
            for label in labels:
                label_id = label['id']
                # only caption arrays from specified samples are saved
                if label_id in data:
                    data[label_id]['captions'] = label['caption']

        # format the input
        pprint(labels[0])

    def _load_features(self):
        folder = join(self._folder, self._dtype, 'feat');
        #file_list = glob(os.path.join(folder, '*.npy'))
        file_list = join(folder, 'xBePrplM4OA_6_18.avi.npy')

        data = np.load(file_list)
        pprint(data)
        print(data.shape)

    def _build_dict(self):
        """
        Scan through the training set and build word-index associations.
        """
        tags = ('<bos>', '<eos>', '<pad>', '<unk>')
        lut = {tag: index for index, tag in enumerate(tags)}

        pprint(lut)

    def __len__(self):
        """
        Returns the size of the dataset.
        """
        return len(self._ids)

    def __getitem__(self, index):
        """
        Support the indexing such that dataset[i] can be used to get ith sample.
        """
        pass
