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
    # (speaker, sentence) association
    sp_se_list = []
    # maximum frames
    n_frames = -1
    for speaker in dataset.speakers:
        for sentence in dataset.sentences:
            r_sp = dataset.data['speaker'] == speaker
            r_se = dataset.data['sentence'] == sentence
            n_rows = dataset.data[r_sp & r_se].shape[0]

            if n_rows > 0:
                sp_se_list.append((speaker, sentence))
                if n_rows > n_frames:
                    n_frames = n_rows

    assert n_frames > 0
    logger.info('Maximum frame count is {}'.format(n_frames))

    n_samples = len(sp_se_list)
    n_features = dataset.x.shape[1]
    # last class is null
    n_classes = len(dataset.lut)+1
    # samples, timesteps (n_frames), features
    xp = np.zeros([n_samples, n_frames, n_features])
    # [0, n_classes), use n_classes as the null class
    yp = np.full([n_samples, n_frames, n_classes], n_classes)

    # unpack instance sets and re-group the features
    for index, (speaker, sentence) in enumerate(sp_se_list):
        r_sp = dataset.data['speaker'] == speaker
        r_se = dataset.data['sentence'] == sentence
        # sort by ascended frame IDs
        r_fr = dataset.data.loc[r_sp & r_se].sort_values('frame')
        n_rows = r_fr.shape[0]
        # write back to the matrix
        xp[index, :n_rows, :] = r_fr.loc[:, 'f0':].values
        yp[index, :n_rows, :] = to_categorical(r_fr['label'].values, n_classes)

    return xp, yp, (n_samples, n_frames, n_features, n_classes)
