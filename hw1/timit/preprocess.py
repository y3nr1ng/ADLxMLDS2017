"""
1) Remove consecutive duplicate labels.
2) Remove leading and trailing silence.
"""
from itertools import groupby

def trim(sentence):
    # remove consecutive labels
    sentence = [x[0] for x in groupby(sentence)]
    # remove leading and trailing silence
    begin = 1 if sentence[0] == 'sil' else 0
    end = -1 if sentence[0] == 'sil' else None
    return sentence[slice(begin, end)]
