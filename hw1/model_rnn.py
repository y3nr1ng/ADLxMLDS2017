from timit import reader
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
import keras
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense

# load the dataset
dataset = reader.TIMIT('data')

start = timer()
dataset.load('train')
end = timer()
print('Data loaded in {0:.3f}s\n'.format(end-start))

# preview
print(dataset.dump(3))

# start training
n_samples, n_features = dataset.x.shape
n_classes = 48

# sample, time steps, features
x_train = np.expand_dims(dataset.x, axis=1)
y_train = keras.utils.to_categorical(dataset.y, n_classes)
y_train = np.expand_dims(y_train, axis=1)

print('Building model...\n')
model = Sequential()
model.add(LSTM(1024, input_shape=(1, n_features), return_sequences=True))
model.add(LSTM(1024, return_sequences=True))
#model.add(Dense(512, activation='relu'))
model.add(TimeDistributed(Dense(n_classes, activation='relu')))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Training started\n')
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2, verbose=1)
