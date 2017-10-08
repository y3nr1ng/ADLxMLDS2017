import os
import csv
import pandas as pd

from .frames import Frames
import pickle
import os
import collections

class TIMIT:
    def __init__(self):
        pass

    def load(self, root, name, model='mfcc', has_label=True):
        """
        Load the TIMIT dataset from specified location.

        Parameters
        ----------
        root: str
            Root directory of the dataset.
        name: str
            Name of the dataset.
        model: str
            Model to use for training or inference.
        has_label: bool, default to True
            Whether the dataset is labeled.
        """
        self._root = root
        self._name = name
        self._model = model

        df_labels = self._load_labels()
        df_features = self._load_features()

        result = pd.merge(df_labels, df_features,
                          on=['speaker', 'sentence', 'frame'])
        return result

    def _load_labels(self):
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
            'label': pd.Categorical(data[3])
        })
        return df

    def _load_features(self):
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
        return df

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
            print('Frame ID is not an integer')
        return [speaker, sentence, frame]

def parsePhoneLUT(raw, sep='\t'):
    p48, _, p39 = raw.strip().partition(sep)
    return p48, p39

def parsePhoneCharPair(raw, sep='\t'):
    phone, index, char = raw.strip().split(sep)
    return phone, (index, char)

def loadMap(folder):
    # load the phone-character map
    mapPath = os.path.join(folder, '48phone_char.map')
    with open(mapPath, 'r') as fd:
        lut = dict(parsePhoneCharPair(line) for line in fd)

    # simplify the mapping
    mapPath = os.path.join(folder, 'phones', '48_39.map')
    with open(mapPath, 'r') as fd:
        phoneLUT = dict(parsePhoneLUT(line) for line in fd)
    for oldPhone in lut:
        newPhone = phoneLUT[oldPhone]
        if oldPhone != newPhone:
            oldMap = lut[oldPhone]
            newMap = lut[newPhone]
            print('Remap {}\'{}\' as {}\'{}\''.format(
                    oldPhone, oldMap, newPhone, newMap))
            newMap = (newMap[0], oldMap[1])
            lut[oldPhone] = newMap

    print()

    return lut

def mergeLabelAndData(labels, data):
    samples = {
        key: list(zip(phones, data[key])) for key, phones in labels.items()
    }
    return samples

def loadTIMIT(folder, name, force=False):
    # phone-character map
    lut = loadMap(folder)
    print('Phone-character mapping')
    for key, value in lut.items():
        print(' {} -> {}'.format(key, value))
    print()

    dataPath = os.path.join(folder, '{}.pkl'.format(name))
    if os.path.isfile(dataPath) and not force:
        print('Use previously saved pickle')
        # file exists and allow to use previous result
        with open(dataPath, 'rb') as fd:
            samples = pickle.load(fd)
            dimension = pickle.load(fd)
    else:
        labels = loadLabel(folder, name, lut)
        data, dimension = loadData(folder, name)

        if len(labels) != len(data):
            raise ValueError('Unable to form label-data pair')
        samples = mergeLabelAndData(labels, data)

        # save to pkl
        with open(dataPath, 'wb') as fd:
            pickle.dump(samples, fd, pickle.HIGHEST_PROTOCOL)
            pickle.dump(dimension, fd, pickle.HIGHEST_PROTOCOL)

    return samples, dimension, lut
