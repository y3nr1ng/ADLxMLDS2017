"""
Train a RNN on the TIMIT dataset.
"""
from timit import reader, process
from timeit import default_timer as timer

# limit TensorFlow to use specific device
import tensorflow as tf
tf_config = tf.ConfigProto(device_count={'GPU': 1})
tf_session = tf.Session(config=tf_config)
# assign the session for Keras
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
K.tensorflow_backend.set_session(tf_session)

import numpy as np
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, TimeDistributed, Dense

import logging
logger = logging.getLogger()

# set the logging format
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)-5s %(message)s',
                              '%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)
# set the log level
logger.setLevel(logging.DEBUG)

# load the dataset
dataset = reader.TIMIT('data')

start = timer()
dataset.load('train')
end = timer()
logger.debug('Data loaded in {0:.3f}s\n'.format(end-start))

# preivew
print(dataset.dump(3))

# preprocess the data to 3-D
x_train, y_train, dimension = process.group_by_sentence(dataset)
(n_sampes, n_timestpes, n_features, n_classes) = dimension

logger.info('Building model...\n')
model = Sequential()
model.add(Bidirectional(LSTM(512, return_sequences=True),
                        input_shape=(n_timestpes, n_features)))
model.add(LSTM(2048, return_sequences=True))
model.add(TimeDistributed(Dense(n_classes, activation='relu')))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

print('Training started\n')
history = model.fit(x_train, y_train,
                    epochs=10, validation_split=0.2, verbose=1)

# serialize model to JSON
model_file = model.to_json()
with open('rnn.json', 'w') as fd:
    fd.write(model_file)
# serialize weights to HDF5
model.save_weights('rnn.h5')
logger.info('Model saved\n')
