from imutils import build_montages
from imutils import paths
import argparse
import random
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('folder', type=str, help='path to input directory of images')
args = ap.parse_args()

# grab the paths to the images, then randomly select a sample of them
imagePaths = list(paths.list_images(args.folder))

# initialize the list of images
images = []

# loop over the list of image paths
for imagePath in imagePaths:
    # load the image and update the list of images
    image = cv2.imread(imagePath)
    images.append(image)

# construct the montages for the images
montages = build_montages(images, (64, 64), (8, 8))

# save the image
cv2.imwrite(os.path.join(args.folder, 'montage.jpg'), montages[0])
