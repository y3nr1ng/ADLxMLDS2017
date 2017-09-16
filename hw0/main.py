import os
import gzip
import csv
import time
import logging
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

# create the logger
logger = logging.getLogger('hw0')
logger.setLevel(logging.DEBUG)
# log format
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
# create console handler
ch = logging.StreamHandler()
ch.setFormatter(formatter)
# add the handler to the logger
logger.addHandler(ch)

def load_mnist_labels(path, kind='train'):
    """ Load MNIST label from `path` """
    labels_path = os.path.join(path, '{}-labels-idx1-ubyte.gz'.format(kind))
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    return labels

def load_mnist_images(path, kind='train', n_images=None):
    """ Load MNIST images from `path` """
    images_path = os.path.join(path, '{}-images-idx3-ubyte.gz'.format(kind))
    with gzip.open(images_path, 'rb') as impath:
        # try to fetch information from the labels
        if not n_images:
            labels = load_mnist_labels(path, kind=kind)
            n_images = len(labels)
        images = np.frombuffer(impath.read(), dtype=np.uint8, offset=16).reshape(n_images, 784)
    return images

def load_mnist(path, kind='train'):
    labels = load_mnist_labels(path, kind=kind)
    images = load_mnist_images(path, kind=kind, n_images=len(labels))

    return images, labels

# load the training data
logger.info('load training set')
X_train, y_train = load_mnist('data', kind='train')

def hist_eq(image, n_bins=196):
    """ Perform histogram equalization on a flattened image array """
    imhist, bins = np.histogram(image, n_bins, normed=True)

    # cumulative distribution function
    cdf = imhist.cumsum()
    # normalize
    cdf = (n_bins-1) * cdf / cdf[-1]

    # use linear interpolation of cdf to find new values
    image_eq = np.interp(image, bin[:-1], cdf)

    return image_eq

# apply histogram equalization to images
X_train = np.apply_along_axis(hist_eq, axis=1, arr=X_train)

# create classifier
classifier = SVC(C=10, kernel='poly')

# execute
logger.info('training START')
start = time.time()
classifier.fit(X_train, y_train)
end = time.time()
logger.info('training END, elapsed {:.2f}s'.format(end-start))

# test the classifier using the original set
score = classifier.score(X_train, y_train)
logger.debug('score = {:.5f}'.format(score))

# load the test data
logger.info('load test set')
X_test = load_mnist_images('data', kind='t10k', n_images=10000)
# predict
logger.info('predict START')
start = time.time()
y_test = classifier.predict(X_test)
end = time.time()
logger.info('predict END, elapsed {:.2f}s'.format(end-start))

# save the result
filename = 'result.csv'
y_index = np.arange(10000)
np.savetxt(filename, np.c_[y_index,y_test], fmt='%d', header='id,label', delimiter=',', comments='')
logger.info('results are saved as "{}"'.format(filename))
