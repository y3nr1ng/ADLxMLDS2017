import logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np

def _default_splitter(string):
    return string.split('\t')

def text_to_onehot(text, splitter=_default_splitter, rule_path='valid_tags.txt'):
    '''Convert texts to one-hot vectors through rule definitions.

    Parameters
    ----------
    text: pd.DataFrame
        text of interest, must have 'id' and 'tags' columns
    rule_path: str
        path to the definition file, one tag per line

    Returns
    -------
    np.ndarray
        array of dimension [batch size, embedding dimension]
    '''
    if not hasattr(text_to_onehot, 'valid_tags'):
        logger.info('loading valid tags from file \'{}\''.format(rule_path))
        with open(rule_path, 'r') as f:
            lines = f.read().splitlines()
        text_to_onehot.valid_tags = {
            tag: index for index, tag in enumerate(lines)
        }

    labels = np.zeros(
        (len(text), len(text_to_onehot.valid_tags)), dtype=np.float32
    )
    for index, (_, row) in enumerate(text.iterrows()):
        tags = splitter(row['tags'])
        for tag in tags:
            labels[index, text_to_onehot.valid_tags[tag]] = 1.0
    return labels
