import os
import gzip
import csv
import time
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

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
print('loading training set')
X_train, y_train = load_mnist('data', kind='train')

# create classifier
n_estimators = 10
classifier = OneVsRestClassifier(
    BaggingClassifier(
        SVC(C=10, kernel='poly'),
        max_samples=1.0/n_estimators,
        n_estimators=n_estimators,
        n_jobs=4
    )
)

# execute
print('training...')
start = time.time()
classifier.fit(X_train, y_train)
end = time.time()
print('elapsed {:.2f}s'.format(end-start))

# test the classifier using the original set
score = classifier.score(X_train, y_train)
print('score={}%%'.format(score*100))

# load the test data
print('loading test set')
X_test = load_mnist_images('data', kind='t10k', n_images=10000)
# predict
print('predicting...')
y_test = classifier.predict(X_test)

# save the result
print('saving the result')
y_index = np.arange(10000)
np.savetxt('result.csv', np.c_[y_index,y_test], fmt='%d', header='id,label', delimiter=',', comments='')
