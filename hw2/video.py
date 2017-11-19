"""
Mask the entire dataset as a module.
"""
from os import listdir
from os.path import isfile, join, exists, basename
from glob import glob
import json
import numpy as np
import re
import pickle
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

        # build word dictionary, forward and inverse
        (self._dict, self._rdict) = self._build_dict()

        #DEBUG
        # 1) load the first item
        # 2) extract the first caption
        # 3) print the vector from
        # 4) convert back to words
        #for _, data in self._data.items():
        #    caption = data['captions'][0]
        #    print(caption)
        #    vector = self.caption_to_vector(caption)
        #    print('forward [{}]'.format(vector))
        #    caption = self.vector_to_caption(vector)
        #    print('reverse [{}]'.format(caption))
        #    break

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
        data = {k: {} for k in self._ids}

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
        def tokenize(sentence):
            return re.compile('\w+').findall(sentence)

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
                        captions = [tokenize(s) for s in label['caption']]
                        data[label_id]['captions'] = captions
                    else:
                        logger.debug('Unused label \'{}\''.format(label_id))
        else:
            logger.warning('Unable to find the label file')
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
            # load array from file
            file_name = basename(file_path)
            features_id = file_name[:file_name.rindex('.npy')]
            features = np.load(file_path)
            # update feature record
            if not hasattr(self, '_n_enc_steps') or not hasattr(self, '_n_features'):
                (self._n_enc_steps, self._n_features) = features.shape
            data[features_id]['features'] = np.load(file_path)
        return data

    def _build_dict(self, name='lut'):
        """
        Scan through the dataset and build word-index associations.
        """
        lut_file = '{}.pkl'.format(name)
        if isfile(lut_file):
            logger.warning('Loading lookup tables from \'{}\''.format(name))
            with open(lut_file, 'rb') as fd:
                lut, rlut = pickle.load(fd)
                self._n_dec_steps = pickle.load(fd)
        else:
            # <bos> begin of sentence
            # <eos> end of sentence
            # <pad> paddings
            # <unk> unknown words
            tags = {'<bos>', '<eos>', '<pad>', '<unk>'}
            self._n_dec_steps = 0
            for _, data in self._data.items():
                for caption in data['captions']:
                    # update tags
                    tags.update(caption)
                    # update decoder steps
                    cap_len = len(caption)
                    if cap_len > self._n_dec_steps:
                        self._n_dec_steps = cap_len
            # convert to lookup table (word -> index)
            lut = {tag: index for index, tag in enumerate(tags)}
            logger.info('{} unique words in the dataset'.format(len(lut)))

            # build inverse lookup table
            rlut = {v:k for k, v in lut.items()}

            # save to pickle
            with open(lut_file, 'wb') as fd:
                pickle.dump((lut, rlut), fd)
                pickle.dump(self._n_dec_steps, fd)
            logger.warning('Lookup tables saved to \'{}\''.format(name))

        return lut, rlut

    def caption_to_vector(self, caption):
        # convert sequence
        unk = self._dict['<unk>']
        print(caption)
        caption = [self._dict.get(w, unk) for w in caption]
        # pad
        vector = np.full(self._n_dec_steps, self._dict['<pad>'])
        vector[:len(caption)] = np.asarray(caption)

        return vector

    def vector_to_caption(self, vector):
        # ignore paddings and convert sequence
        pad = self._dict['<pad>']
        caption = [self._rdict[v] for v in vector if v != pad]

        return caption

    def generator(self):
        pad = self._dict['<pad>']
        bos = self._dict['<bos>']
        eos = self._dict['<eos>']
        unk = self._dict['<unk>']

        n_timesteps = self._n_enc_steps + self._n_dec_steps
        for vid in self._ids:
            data = self._data[vid]

            x_enc = np.full((1, n_timesteps, self._n_features),
                            0.0, dtype=np.float32)
            x_enc[0, :self._n_enc_steps, :] = data['features']
            for caption in data['captions']:
                x_dec = np.full((1, n_timesteps, len(self._dict)),
                                0.0, dtype=np.float32)
                y = np.full((1, n_timesteps, len(self._dict)),
                             0.0, dtype=np.float32)

                # fill the paddings
                x_dec[0, :self._n_enc_steps, pad] = 1.0
                y[0, :self._n_enc_steps, pad] = 1.0
                # fill the <bos>
                x_dec[0, self._n_enc_steps, bos] = 1.0
                # fill the <eos>
                y[0, n_timesteps-1, eos] = 1.0
                # fill the sentence
                for i, w in enumerate(caption):
                    v = self._dict.get(w, unk)
                    x_dec[0, (self._n_enc_steps+1)+i, v] = 1.0
                    y[0, self._n_enc_steps+i, v] = 1.0

                # send the result back
                yield ({'input_1': x_enc, 'input_2': x_dec}, {'dense_1': y})

    @property
    def shape(self):
        """
        Shape of the dataset.

        Return
        ------
        (n_features, n_words, n_timesteps)
            n_features: int
                Number of extracted features per frame
            n_words: int
                Number of extracted words from provided captions
            n_timesteps: int
                Timesteps per sample.
        """
        n_timesteps = self._n_enc_steps + self._n_dec_steps
        return (self._n_features, len(self._dict), n_timesteps)

    def __len__(self):
        n_samples = 0
        for _, data in self._data.items():
            n_samples += len(data['captions'])
        return n_samples
