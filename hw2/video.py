"""
Mask the entire dataset as a module.
"""
from os import listdir
from os.path import isfile, join, exists, basename
from glob import glob
import json
import numpy as np
from timeit import default_timer as timer
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
        """
        List wanted IDs from the *_id.txt file.

        Return
        ------
        A list of designated IDs, noted that .avi extensions are not removed.
        """
        # generate id file name
        prefix = self._dtype
        prefix = prefix[:-5] if self._dtype.endswith('_data') else prefix
        id_file = join(self._folder, '{}_id.txt'.format(prefix))
        # load ids from file if exists
        if isfile(id_file):
            with open(id_file, 'r') as fd:
                ids = [line.rstrip('\n') for line in fd]
        else:
            folder = join(self._folder, self._dtype, 'feat')
            file_list = [f for f in listdir(folder) if isfile(join(folder, f))]
            try:
                # remove the file extension
                ids = list(map(lambda s: s[:s.rindex('.npy')], file_list))
            except ValueError:
                logger.error('Invalid filename exists')
        logger.info('{} IDs found'.format(len(ids)))
        return ids

    def _load_data(self):
        # create empty dataset, nested dict
        data = dict.fromkeys(self._ids, {'captions': [], 'features': []})

        start = timer()
        data = self._load_labels(data)
        data = self._load_features(data)
        end = timer()
        logger.info('Data loaded in {:.3f}s'.format(end-start))

        return data

    def _load_labels(self, data):
        """
        Load the dictionary representation of labels.

        Parameter
        ---------
        data: dict
            Dictionary with IDs as keys.

        Return
        ------
        Dataset loaded with labels (if the file exists).
        """
        # generate label file name
        prefix = self._dtype
        prefix = prefix[:-5] if self._dtype.endswith('_data') else prefix
        label_file = join(self._folder, '{}_label.json'.format(prefix))
        # load ids from file if exists
        if isfile(label_file):
            with open(label_file, 'r') as fd:
                for label in json.load(fd):
                    label_id = label['id']
                    # only caption arrays from specified samples are saved
                    if label_id in data:
                        data[label_id]['captions'] = label['caption']
        return data

    def _load_features(self, data):
        """
        Load frame features from individual .npy files.

        Parameter
        ---------
        data: dict
            Dictionary with IDs as keys.

        Return
        ------
        Dataset loaded with features.
        """
        folder = join(self._folder, self._dtype, 'feat')
        for file_path in glob(join(folder, '*.npy')):
            file_name = basename(file_path)
            features_id = file_name[:file_name.rindex('.npy')]
            data[features_id]['features'] = np.load(file_path)
        return data

    def _build_dict(self):
        """
        Scan through the training set and build word-index associations.
        """
        tags = ('<bos>', '<eos>', '<pad>', '<unk>')
        lut = {tag: index for index, tag in enumerate(tags)}

        pprint(lut)

        raise RuntimeError('_build_dict')

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
