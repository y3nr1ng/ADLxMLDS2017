from timit import reader
from timeit import default_timer as timer

# limit TensorFlow to use specific device
import tensorflow as tf
tf_config = tf.ConfigProto(device_count={'CPU': 4})
tf_session = tf.Session(config=tf_config)
# assign the session for Keras
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
K.tensorflow_backend.set_session(tf_session)

import numpy as np
from keras.models import Sequential
from keras.layers import GRU, TimeDistributed, Dense

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

# sample, time steps, features
x_train = np.expand_dims(dataset.x, axis=1)
y_train = np.expand_dims(np.expand_dims(dataset.y, axis=1), axis=2)

print('Building model...\n')
model = Sequential()
model.add(GRU(n_features, input_shape=(1, n_features), return_sequences=True))
model.add(GRU(n_features, input_shape=(1, n_features), return_sequences=True))
model.add(TimeDistributed(Dense(1, activation='relu')))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
print(model.summary())

print('Training started\n')
model.fit(x_train, y_train, epochs=5, validation_split=0.2)
