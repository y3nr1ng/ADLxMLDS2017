"""
1) Load the TIMIT dataset.
2) Remap the phones to english characters (48->48).
3) Remap the phones (48->39).
"""

import os
import collections
from .frames import Frames

def parseInstanceID(raw, sep='_'):
    speaker, sentence, frame = raw.strip().split(sep)
    return (speaker, sentence), int(frame)

def parseData(raw, sep='\t'):
    instance, data = raw.strip().partition(sep)
    instance, frame = parseInstanceID(instance)
    return instance, frame, data

def loadData(folder, name):
    dataName = '{}.ark'.format(name)
    dataPath = os.path.join(folder, 'mfcc', dataName)
    with open(dataPath, 'r') as fd:
        # use list() as the initializer function for missing keys
        data = collections.defaultdict(Frames)
        for line in fd:
            instance, frame, datum = parseData(line)
            # raw frame ID starts with 1
            frame -= 1
            # store the label at specified frame position
            frameList = data[instance]
            frameList[frame] = datum
            data[instance] = frameList
    return data

def parseLabel(raw, sep=','):
    instance, label = raw.strip().split(sep)
    instance, frame = parseInstanceID(instance)
    return instance, frame, label

def loadLabel(folder, name):
    labelName = '{}.lab'.format(name)
    labelPath = os.path.join(folder, 'label', labelName)
    with open(labelPath, 'r') as fd:
        # use list() as the initializer function for missing keys
        labels = collections.defaultdict(Frames)
        for line in fd:
            instance, frame, label = parseLabel(line)
            # raw frame ID starts with 1
            frame -= 1
            # store the label at specified frame position
            frameList = labels[instance]
            frameList[frame] = label
            labels[instance] = frameList
    return labels

def parsePhoneLUT(raw, sep='\t'):
    p48, _, p39 = raw.strip().partition(sep)
    return p48, p39

def parsePhoneCharPair(raw, sep='\t'):
    phone, _, char = raw.strip().split(sep)
    return phone, char

def loadMap(folder):
    # load the phone-character map
    mapPath = os.path.join(folder, '48phone_char.map')
    with open(mapPath, 'r') as fd:
        phone2char = dict(parsePhoneCharPair(line) for line in fd)

    # simplify the mapping
    mapPath = os.path.join(folder, 'phones', '48_39.map')
    with open(mapPath, 'r') as fd:
        phoneLUT = dict(parsePhoneLUT(line) for line in fd)
    for oldPhone in phone2char:
        newPhone = phoneLUT[oldPhone]
        if oldPhone != newPhone:
            phone2char[oldPhone] = phone2char[newPhone]
            print('remap \'{}\' as \'{}\''.format(oldPhone, newPhone))
    print()

    # dump the map
    print('phone-character mapping')
    for key, value in phone2char.items():
        print(' {} -> {}'.format(key, value))
    print()

    return phone2char

def loadTIMIT(folder, name):
    pass
