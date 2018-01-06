import argparse
import os
from os.path import isfile
import re

import pandas as pd
import numpy as np

def list_folders(root_dir, has_root=True):
    '''Retrieve all the folders in specified location.'''
    dir_list = [
        d for d in os.listdir(root_dir) if not isfile(os.path.join(root_dir, d))
    ]
    if not has_root:
        root_dir = ''
    return [os.path.join(root_dir, d) for d in dir_list]

def generate_csv(root_dir):
    '''Extract the G/D loss from folder names.'''
    df = pd.DataFrame(columns=['epochs', 'g_loss', 'd_loss'])
    dir_list = list_folders(root_dir, has_root=False)
    pattern = re.compile('^(\d+)_g([\.|\d]*)_d([\.|\d]*)$')
    for i, d in enumerate(dir_list):
        epochs, g_loss, d_loss = pattern.findall(d)[0]
        df.loc[i] = [int(epochs), float(g_loss), float(d_loss)]
    df = df.sort_values(by='epochs')
    df.to_csv('{}.csv'.format(root_dir), index=False)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    dir_list = list_folders(args.root)
    for d in dir_list:
        generate_csv(d)
