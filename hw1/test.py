from timit import reader, preprocess
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

testSize = 0.25
batchSize = 16

samples, dimension, lut = reader.loadTIMIT('data', 'train', force=True)

# determine train set and test set for validations
instances = [k for k in samples.keys()]
datasets = train_test_split(instances, test_size=testSize)
print('Split train and test set, test size is {}'.format(testSize))
(x_train, y_train) = (datasets[0], [samples[x] for x in datasets[0]])
(x_test, y_test) = (datasets[1], [samples[x] for x in datasets[1]])

print('Building model...')
model = Sequential()
model.add(Embedding(dimension, dimension))
model.add(LSTM(48, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Training started')
model.fit(x_train, y_train,
          batch_size=batchSize, epochs=10,
          validation_data=(x_test, y_test))
score, accuracy = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Score = ', score)
print('Accuracy = ', accuracy)
