"""
Train a RNN on the TIMIT dataset.
"""
from timit import reader, process
from timeit import default_timer as timer

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# limit TensorFlow to specific device
tf_config = tf.ConfigProto(device_count={'GPU': 1})
tf_session = tf.Session(config=tf_config)
# assign the session for Keras
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
K.tensorflow_backend.set_session(tf_session)

import numpy as np
from keras.models import model_from_json, Sequential
from keras.layers import Masking, Bidirectional, LSTM, TimeDistributed, Dense
from keras.initializers import RandomUniform
from keras.optimizers import SGD
from keras import metrics

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

def load_dataset(name, folder='data', has_label=True):
    dataset = reader.TIMIT(folder)
    # load the raw data
    start = timer()
    dataset.load(name, has_label=has_label)
    end = timer()
    logger.debug('Data loaded in {0:.3f}s\n'.format(end-start))

    print(dataset.dump(3))

    return dataset

def build_model(dimension):
    # unpack specification from the dataset
    (n_timesteps, n_features, n_classes) = dimension
    input_shape = (n_timesteps, n_features)

    logger.info('Building model...')
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=input_shape))
    model.add(Bidirectional(LSTM(64,
                                 dropout=0.01, recurrent_dropout=0.01,
                                 return_sequences=True),
                            input_shape=input_shape))
    model.add(TimeDistributed(Dense(n_classes, activation='softmax')))
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=[metrics.categorical_accuracy])
    return model

def start_training(model, x, y, batch_size=32, epochs=10, validation_split=0.2):
    logger.info('Training started')
    history = model.fit(x, y, validation_split=validation_split,
                        epochs=epochs, batch_size=batch_size, verbose=1)

    scores = model.evaluate(x, y, verbose=1)
    logger.info('{}: {:.2f}%'.format(model.metrics_names[1], scores[1]*100))

    return model, history

def save_model(model, name='rnn'):
    # serialize model
    model_file = model.to_json()
    with open('{}.json'.format(name), 'w') as fd:
        fd.write(model_file)
    # serialize weights
    model.save_weights('{}.h5'.format(name))

    logger.info('Model saved as \'{}\''.format(name))

def load_model(name='rnn'):
    # load model
    with open('{}.json'.format(name), 'r') as fd:
        model = model_from_json(fd.read())
    # retrieve the dimensions
    (_, n_timesteps, n_features) = model.layers[0].input_shape

    # load weights
    model.load_weights('{}.h5'.format(name))
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=[metrics.categorical_accuracy])

    logger.info('Model loaded from \'{}\''.format(name))
    return model, (n_timesteps, n_features)

def to_sequence(dataset, y):
    # remove consecutive elements
    y = y[np.insert(np.diff(y).astype(np.bool), 0, True)]
    # convert to characters
    yc = ''.join([list(dataset.lut.values())[i] for i in y])
    return yc

if __name__ == '__main__':
    TRAIN_MODEL = True
    SAVE_MODEL = True
    DATASET_NAME = 'train'
    HAS_LABEL = True

    dataset = load_dataset(DATASET_NAME, has_label=HAS_LABEL)

    if TRAIN_MODEL:
        x, y, dimension = process.group_by_sentence(dataset)

        model = build_model(dimension)
        model, history = start_training(model, x, y, epochs=50)

        if SAVE_MODEL:
            save_model(model)
    else:
        model, dimension = load_model()
        # restrict the output by model shape
        x, y, dimension = process.group_by_sentence(dataset, dimension)

    y_predict = model.predict(x, verbose=2)
    # convert from categorical to continuous
    y_predict = np.argmax(y_predict, axis=2)

    #DEBUG
    y = np.argmax(y, axis=2)

    # translate the prediction result
    for i in range(len(dataset.instances)):
        print('pred [{}]'.format(to_sequence(dataset, y_predict[i, :])))
        print('trut [{}]'.format(to_sequence(dataset, y[i, :])))
        print()
