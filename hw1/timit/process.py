import numpy as np
from keras.utils import to_categorical

from itertools import groupby

import logging
logger = logging.getLogger()

def group_by_sentence(dataset, dimension=None):
    """
    Convert DataFrame into sentence-based 3-D NumPy array.

    Parameters
    ----------
    dataset: TIMIT
        Dataset that contains parsed TIMIT data.
    """
    # maximum frames
    if dimension:
        n_frames = dimension[0]
    else:
        n_frames = dataset.data['frame'].max()
    logger.info('Maximum frame count is {}'.format(n_frames))

    # (speaker, sentence) association
    logger.info('{} speakers, {} sentences'.format(len(dataset.speakers),
                                                   len(dataset.sentences)))
    n_samples = len(dataset.instances)
    logger.info('{} available sentence (set) samples'.format(n_samples))

    if dimension:
        n_features = dimension[1]
    else:
        n_features = dataset.x.shape[1]
    # last class is null
    n_classes = len(dataset.lut)
    # samples, timesteps (n_frames), features
    xp = np.full([n_samples, n_frames, n_features], 0)
    has_yp = 'label' in dataset.data
    if has_yp:
        # [0, n_classes), use n_classes as the null class
        yp = np.full([n_samples, n_frames, n_classes], -1)
    else:
        yp = None

    # unpack instance sets and re-group the features
    for index, (speaker, sentence) in enumerate(dataset.instances):
        r_sp = dataset.data['speaker'] == speaker
        r_se = dataset.data['sentence'] == sentence
        # sort by ascended frame IDs
        r_fr = dataset.data.loc[r_sp & r_se].sort_values('frame')
        n_rows = r_fr.shape[0]
        # write back to the matrix
        xp[index, :n_rows, :] = r_fr.loc[:, 'f0':].values
        if has_yp:
            yp[index, :n_rows, :] = to_categorical(r_fr['label'].values, n_classes)

    return xp, yp, (n_frames, n_features, n_classes)

def to_sequence(dataset, y):
    # convert to characters
    yc = [list(dataset.lut.values())[i] for i in y]
    # remove consecutive elements and join as a string sequence
    yc = [i for i, _ in groupby(yc)]
    # remove leading and trailing silence
    sil = dataset.lut['sil']
    begin = 1 if yc[0] == sil else 0
    end = -1 if yc[-1] == sil else None
    yc[:] = yc[slice(begin, end)]
    # return the string sequence
    return ''.join(yc)

def edit_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1)+1)
    for i2, c2 in enumerate(s2):
        _distances = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                _distances.append(distances[i1])
            else:
                _distances.append(1+min((distances[i1], distances[i1+1], _distances[-1])))
        distances = _distances
    return distances[-1]
