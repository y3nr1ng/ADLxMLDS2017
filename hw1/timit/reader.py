import pandas as pd
import os
import csv
import collections

import logging
logger = logging.getLogger()

class TIMIT:
    def __init__(self, root):
        """
        Parameters
        ----------
        root: str
            Root directory of the dataset.
        """
        if root is None:
            raise ValueError('Invalid root directory')
        self._root = root
        self.lut = self._load_map()
        self.data = []

    def _load_map(self):
        # load the phone-character map
        def phone_to_char(raw, sep='\t'):
            phone, _, char = raw.strip().split(sep)
            return phone, char
        path = os.path.join(self._root, '48phone_char.map')
        with open(path, 'r') as fd:
            lut = collections.OrderedDict(phone_to_char(line) for line in fd)

        # simplify the mapping
        def phone_remap(raw, sep='\t'):
            p48, _, p39 = raw.strip().partition(sep)
            return p48, p39
        path = os.path.join(self._root, 'phones', '48_39.map')
        with open(path, 'r') as fd:
            remap_lut = dict(phone_remap(line) for line in fd)
        for old_phone in remap_lut:
            new_phone = remap_lut[old_phone]
            if old_phone != new_phone:
                lut[old_phone] = lut[new_phone]
                logger.info('\'{}\' -> \'{}\''.format(old_phone, new_phone))

        return lut

    def load(self, name, model='mfcc', has_label=True):
        """
        Load the TIMIT dataset from specified location.

        Parameters
        ----------
        name: str
            Name of the dataset.
        model: str
            Model to use for training or inference.
        has_label: bool, default to True
            Whether the dataset is labeled.
        """
        self._name = name
        self._model = model

        df_labels = self._load_labels()
        df_features = self._load_features()

        self.data = pd.merge(df_labels, df_features,
                             on=['speaker', 'sentence', 'frame'])

    def _load_labels(self):
        """ Load .lab file of specified model in DataFrame. """
        src_dir = os.path.join(self._root, 'label')
        filename = '{}.lab'.format(self._name)
        path = os.path.join(src_dir, filename)
        with open(path, 'r') as fd:
            data = []
            for line in fd:
                instance, label = line.strip().split(',')
                instance = TIMIT.parse_instance_id(instance)
                data.append(instance + [label])
        # convert to column-wise
        data = list(zip(*data))
        df = pd.DataFrame({
            'speaker': data[0], 'sentence': data[1], 'frame': data[2],
            'label': pd.Categorical([list(self.lut.keys()).index(x) for x in data[3]])
        })
        df = TIMIT.instances_as_category(df, ['speaker', 'sentence'])
        return df

    def _load_features(self):
        """ Load .ark file of specified model in DataFrame. """
        src_dir = os.path.join(self._root, self._model)
        filename = '{}.ark'.format(self._name)
        path = os.path.join(src_dir, filename)
        with open(path, 'r') as fd:
            data = []
            for line in fd:
                instance, _, features = line.strip().partition(' ')
                instance = TIMIT.parse_instance_id(instance)
                features = [float(x) for x in features.split()]
                data.append(instance + features)
        # generate feature labels
        n_feature = len(data[0])-3;
        headers = ['f{}'.format(x) for x in range(n_feature)]
        headers = ['speaker', 'sentence', 'frame'] + headers;
        df = pd.DataFrame(data, columns=headers)
        df = TIMIT.instances_as_category(df, ['speaker', 'sentence'])
        return df

    @staticmethod
    def instances_as_category(df, col):
        """
        Convert list of columns to pandas.Categorical

        Parameters
        ----------
        df: pandas.DataFrame
            DataFrame source.
        col: list(str)
            List of column names to convert .
        """
        for c in col:
            df[c] = df[c].astype('category')
        return df

    def _identify_dimensions(self):
        pass

    @staticmethod
    def parse_instance_id(text, sep='_'):
        """
        Strip the instance ID into speaker, sentence and frame.

        Parameters
        ----------
        text: str
            The raw intance ID.
        sep: str, default to '_'
            Separator used between IDs.
        """
        speaker, sentence, frame = text.strip().split(sep)
        try:
            frame = int(frame)
        except ValueError:
            logger.error('Frame ID is not an integer')
        return [speaker, sentence, frame]

    @property
    def x(self):
        """
        Return all the features (f_i) as an numpy array.
        """
        return self.data.loc[:, 'f0':].values

    @property
    def y(self):
        """
        Return the label column as numpy array.
        """
        return self.data['label'].values

    @property
    def speakers(self):
        """
        Return a list of speaker IDs.
        """
        return self.data['speaker'].cat.categories

    @property
    def sentences(self):
        """
        Return a list of sentence IDs.
        """
        return self.data['sentence'].cat.categories

    def dump(self, head=None):
        """
        Dump the data table.

        Parameters
        ----------
        head: int
            Dump only the first few rows, otherwise, all of them.
        """
        print(self.data.head(head))
