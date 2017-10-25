import numpy as np
from keras.utils import to_categorical

import logging
logger = logging.getLogger()

def group_by_sentence(dataset):
    """
    Convert DataFrame into sentence-based 3-D NumPy array.

    Parameters
    ----------
    dataset: TIMIT
        Dataset that contains parsed TIMIT data.
    """
    # maximum frames
    n_frames = dataset.data['frame'].max()
    logger.info('Maximum frame count is {}'.format(n_frames))

    # (speaker, sentence) association
    logger.info('{} speakers, {} sentences'.format(len(dataset.speakers),
                                                   len(dataset.sentences)))
    sp_se_list = []
    for speaker in dataset.speakers:
        r_sp = dataset.data[dataset.data['speaker'] == speaker]
        for sentence in dataset.sentences:
            r_se = r_sp[r_sp['sentence'] == sentence]
            if not r_se.empty:
                logger.debug('({}, {})'.format(speaker, sentence))
                sp_se_list.append((speaker, sentence))
    n_samples = len(sp_se_list)
    logger.info('{} available sentence (set) samples'.format(n_samples))

    n_features = dataset.x.shape[1]
    # last class is null
    n_classes = len(dataset.lut)+1
    # samples, timesteps (n_frames), features
    xp = np.zeros([n_samples, n_frames, n_features])
    # [0, n_classes), use n_classes as the null class
    yp = np.full([n_samples, n_frames, n_classes], n_classes)

    # unpack instance sets and re-group the features
    for index, (speaker, sentence) in enumerate(sp_se_list):
        logger.debug(index)
        r_sp = dataset.data['speaker'] == speaker
        r_se = dataset.data['sentence'] == sentence
        # sort by ascended frame IDs
        r_fr = dataset.data.loc[r_sp & r_se].sort_values('frame')
        n_rows = r_fr.shape[0]
        # write back to the matrix
        xp[index, :n_rows, :] = r_fr.loc[:, 'f0':].values
        yp[index, :n_rows, :] = to_categorical(r_fr['label'].values, n_classes)

    return xp, yp, (n_samples, n_frames, n_features, n_classes)
