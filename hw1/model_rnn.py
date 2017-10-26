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
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, TimeDistributed, Dense, GRU
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

logger.info('Building model...')
model = Sequential()
model.add(Bidirectional(LSTM(256, return_sequences=True,
                             kernel_initializer=RandomUniform(minval=-0.1, maxval=0.1)),
                        input_shape=(n_timestpes, n_features)))
model.add(TimeDistributed(Dense(n_classes, activation='softmax')))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=[metrics.categorical_accuracy])

batch_size = 1

logger.info('Training started')
history = model.fit(x_train, y_train,
                    epochs=15, batch_size=batch_size, verbose=1)
scores = model.evaluate(x_train, y_train, verbose=1)
logger.info('{}: {:.2f}%'.format(model.metrics_names[1], scores[1]*100))

y_predict = model.predict(x_train, batch_size=batch_size, verbose=1)
print(y_predict)

print('predict')
print(np.argmax(y_predict, axis=2))
print()
print('ground truth')
print(np.argmax(y_train, axis=2))

# serialize model to JSON
model_file = model.to_json()
with open('rnn.json', 'w') as fd:
    fd.write(model_file)
# serialize weights to HDF5
model.save_weights('rnn.h5')
logger.info('Model saved')
