from .frames import Frames
import pickle
import os
import collections

def parseInstanceID(raw, sep='_'):
    """
    Strip the instance ID into three parts:
    1) Speaker ID
    2) Sentence ID
    3) Frame ID
    where speaker ID and sentence ID will combine into a tuple to use as the key
    for dictionaries, and frame ID is converted into integer automatically.

    Parameters
    ----------
    raw: str
        the raw instance ID
    sep: str, default to '_'
        separator between the IDs
    """
    speaker, sentence, frame = raw.strip().split(sep)
    return (speaker, sentence), int(frame)

def parseData(raw, sep=' '):
    instance, _, data = raw.strip().partition(sep)
    instance, frame = parseInstanceID(instance)
    data = [float (x) for x in data.split()]
    return instance, frame, data

def loadData(folder, name):
    dataName = '{}.ark'.format(name)
    dataPath = os.path.join(folder, 'mfcc', dataName)
    dimension = -1
    with open(dataPath, 'r') as fd:
        # use list() as the initializer function for missing keys
        data = collections.defaultdict(Frames)
        index = 0
        for line in fd:
            instance, frame, features = parseData(line)
            # save feature size
            if dimension < 0:
                dimension = len(features)
            # raw frame ID starts with 1
            frame -= 1
            # store the label at specified frame position
            frameList = data[instance]
            frameList[frame] = features
            data[instance] = frameList

            index += 1
            print('\rReading entry {}'.format(index), end='')
        print('\r{} datasets loaded'.format(len(data)))
    return data, dimension

def parseLabel(raw, sep=','):
    instance, label = raw.strip().split(sep)
    instance, frame = parseInstanceID(instance)
    return instance, frame, label

def loadLabel(folder, name, lut):
    labelName = '{}.lab'.format(name)
    labelPath = os.path.join(folder, 'label', labelName)
    with open(labelPath, 'r') as fd:
        # use list() as the initializer function for missing keys
        labels = collections.defaultdict(Frames)
        index = 0
        for line in fd:
            instance, frame, label = parseLabel(line)
            # raw frame ID starts with 1
            frame -= 1
            # store the label at specified frame position
            frameList = labels[instance]
            frameList[frame] = lut[label][0]
            labels[instance] = frameList

            index += 1
            print('\rReading entry {}'.format(index), end='')
        print('\r{} label-sets loaded'.format(len(labels)))
    return labels

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
