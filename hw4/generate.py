import argparse
import os
import logging
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(levelname).1s %(asctime)s [%(name)s] %(message)s', '%H:%M:%S'
)
handler.setFormatter(formatter)
logging.basicConfig(level=logging.DEBUG, handlers=[handler])
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np

import data, model
from gan import WassersteinGAN
import utils

def _test_text_splitter(string, span=2):
    '''Split every nth space.'''
    tags = string.split()
    return [' '.join(tags[i:i+span]) for i in range(0, len(tags), span)]

def load_labels(label_path):
    df = pd.read_csv(label_path, index_col=0, names=['id', 'tags'])
    return utils.text_to_onehot(df, splitter=_test_text_splitter)

def parse_arguments():
    parser = argparse.ArgumentParser('HW4')
    parser.add_argument('label_path', type=str, help='label source')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument(
        '-n', type=int, default=5, help='number of images per label'
    )
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    data_sampler = data.DataSampler()
    noise_sampler = data.NoiseSampler(seed=args.seed)
    g_net = model.Generator()
    d_net = model.Discriminator()

    wgan = WassersteinGAN(g_net, d_net, data_sampler, noise_sampler)
    wgan.restore()

    labels = load_labels(args.label_path)
    for i in range(labels.shape[0]):
        prefix = 'sample_{}'.format(i)
        logger.info('generate images for \'{}\''.format(prefix))
        bx = wgan.generate(labels[i, ...])
        images = data_sampler.to_images(bx)
        for j in range(images.shape[0]):
            skimage.io.imsave(
                os.path.join('samples', '{}_{}.jpg'.format(prefix, j)),
                images[j, ...]
            )
