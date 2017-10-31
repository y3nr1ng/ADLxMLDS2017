from video import Video

import os
# set TensorFlow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# set Keras backend
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.models import model_from_json, Sequential

import argparse
from os import rename
from os.path import exists, getmtime
from datetime import datetime as dt

import logging
logger = logging.getLogger()

# set the logging format
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)-5s %(message)s',
                              '%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)

# set the global log level
logger.setLevel(logging.DEBUG)

def build():
    print('BUILDING')

def train():
    pass

def evaluate():
    pass

def load(name, strict=False):
    """
    Load the latest model.

    Parameters
    ----------
    name: str
        name of the model file
    strict: bool, default to False
        set to True if the model must exists
    """
    model_file = '{}.json'.format(name)
    weight_file = '{}.h5'.format(name)
    if not exists(model_file) or not exists(weight_file):
        if strict:
            raise IOError('Model file not found')
        else:
            return None
    else:
        # load the model
        with open(model_file, 'r') as fd:
            model = model_from_json(fd.read())
        # load weights
        model.load_weights(weight_file)
        #TODO compile
        # model.compile(...)

        logger.info('Model \'{}\' loaded'.format(name))
        return model

def archive(name):
    model_file = '{}.json'.format(name)
    weight_file = '{}.h5'.format(name)
    for f in [model_file, weight_file]:
        if exists(f):
            timestamp = dt.fromtimestamp(getmtime(f)).strftime('%Y%m%d-%H%M%S')
            rename(f, '{}_{}.json'.format(name, timestamp))
    logger.info('Archived last active model')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Video captioning using a sequence-to-sequence model.')
    parser.add_argument('folder', type=str, help='folder of the datasets')
    parser.add_argument('dataset', choices=['train', 'test', 'review'],
                        help='name of the dataset to use')
    parser.add_argument('--output', '-o', type=str, default='output.txt',
                        help='filename of the result')
    parser.add_argument('--mode', '-m', choices=['train', 'infer'],
                        default='infer', help='mode of operation')
    parser.add_argument('--dry', '-d', action='store_true',
                        help='dry run only, the model is not saved')
    parser.add_argument('--reuse', '-r', action='store_true',
                        help='train upon existing model if exists')
    args = parser.parse_args()

    # fine tune the parameters
    model_name = 's2s'

    if args.mode == 'train':
        if not args.reuse:
            archive(model_name)
        model = load(model_name)
        # build the model from scratch if nothing is loaded
        if not model:
            model = build()
    elif args.mode == 'infer':
        model = load(model_name, strict=True)

    data = Video(args.folder, dtype=args.dataset)
