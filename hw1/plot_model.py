"""
Plot the Keras model as an image.
"""
from keras.models import model_from_json
from keras.utils import plot_model

MODEL_NAME = 'rnn_mfcc'

with open('{}.json'.format(MODEL_NAME), 'r') as fd:
    model = model_from_json(fd.read())

plot_model(model, to_file='{}.png'.format(MODEL_NAME),
           show_shapes=True, show_layer_names=True)
