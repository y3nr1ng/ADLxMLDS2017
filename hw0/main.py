import os
import gzip
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def load_mnist_labels(path, kind='train'):
    """ Load MNIST label from `path` """
    labels_path = os.path.join(path, '{}-labels-idx1-ubyte.gz'.format(kind))
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    return labels

def load_mnist_images(path, kind='train', n_images=None):
    """ Load MNIST images from `path` """
    images_path = os.path.join(path, '{}-images-idx3-ubyte.gz'.format(kind))
    with gzip.open(labels_path, 'rb') as lbpath:
        # try to fetch information from the labels
        if not n_images:
            labels = load_mnist_labels(path, kind=kind)
            n_images = len(labels)
        images = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=16)
                   .reshape(len(labels), 784)
    return images

def load_mnist(path, kind='train'):
    labels = load_mnist_labels(path, kind=kind)
    images = load_mnist_images(path, kind=kind, n_images=len(labels))

    return images, labels

# load the training data
X_train, y_train = load_mnist('data', kind='train')

# create classifier
classifier = RandomForestClassifier(criterion='gini', max_depth=100, n_estimators=10)
# execute
classifier.fit(X_train, y_train)

# test the classifier using the original set
score = classifier.score(X_train, y_train)
print('score={}%%'.format(score*100))


    y_test = classifier.predict(X_train)
