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
from keras.layers import Masking, Bidirectional, LSTM, TimeDistributed, Dense, Conv1D
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

def load_dataset(name, folder='data', model='mfcc', has_label=True):
    dataset = reader.TIMIT(folder)
    # load the raw data
    start = timer()
    dataset.load(name, model=model, has_label=has_label)
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
    #model.add(Masking(mask_value=0, input_shape=input_shape))
    model.add(Conv1D(128, 8, padding='same', activation='relu', input_shape=input_shape))
    model.add(Bidirectional(LSTM(128,
                                 dropout=0.2, recurrent_dropout=0.2,
                                 return_sequences=True)))
    model.add(Bidirectional(LSTM(64,
                                 dropout=0.2, recurrent_dropout=0.2,
                                 return_sequences=True)))
    model.add(TimeDistributed(Dense(n_classes, activation='softmax')))
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=[metrics.categorical_accuracy])
    return model

def train(model, x, y, batch_size=32, epochs=10, validation_split=0.1):
    logger.info('Training started')
    history = model.fit(x, y, validation_split=validation_split,
                        epochs=epochs, batch_size=batch_size, verbose=1)

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

if __name__ == '__main__':
    TRAIN_MODEL = False
    REUSE_MODEL = False
    SAVE_MODEL = True
    DATASET_NAME = 'test'
    HAS_LABEL = False

    MODEL_TYPE = 'mfcc'
    MODEL_NAME = 'cnn_{}'.format(MODEL_TYPE)

    dataset = load_dataset(DATASET_NAME, model=MODEL_TYPE, has_label=HAS_LABEL)

    if TRAIN_MODEL:
        x, y, dimension = process.group_by_sentence(dataset)

        if REUSE_MODEL:
            model, dimension = load_model(name=MODEL_NAME)
        else:
            model = build_model(dimension)
        model, history = train(model, x, y, batch_size=64, epochs=100)

        if SAVE_MODEL:
            save_model(model, name=MODEL_NAME)
    else:
        model, dimension = load_model(name=MODEL_NAME)
        # restrict the output by model shape
        x, y, dimension = process.group_by_sentence(dataset, dimension)

    yp = model.predict(x, batch_size=256, verbose=2)
    yp = np.argmax(yp, axis=2)

    n_samples = len(dataset.instances)
    if HAS_LABEL:
        scores = model.evaluate(x, y, batch_size=256, verbose=1)
        logger.info('{}: {:.2f}%'.format(model.metrics_names[1], scores[1]*100))

        # convert back to continuous categoy
        y = np.argmax(y, axis=2)

        avg_edit_dist = 0
        for i in range(n_samples):
            sp = process.to_sequence(dataset, yp[i, :])
            st = process.to_sequence(dataset, y[i, :])
            avg_edit_dist += process.edit_distance(sp, st)
        avg_edit_dist /= n_samples
        logger.info('Average edit distance = {:.3f}'.format(avg_edit_dist))
    else:
        with open('result.csv', 'w') as fd:
            fd.write('id,phone_sequence\n')
            for i, instance in enumerate(dataset.instances):
                instance_id = '{}_{}'.format(instance[0], instance[1])
                logger.debug('[{}] {}'.format(i, instance_id))
                sp = process.to_sequence(dataset, yp[i, :])
                fd.write('{},{}\n'.format(instance_id, sp))
