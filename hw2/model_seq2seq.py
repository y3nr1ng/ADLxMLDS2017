from video import Video

import argparse
from datetime import datetime

import logging
logger = logging.getLogger()

# set the logging format
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(levelname)-5s %(message)s',
                              '%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)

# set the global log level
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Video captioning using a sequence-to-sequence model.')
    parser.add_argument('folder', type=str, help='folder of the datasets')
    parser.add_argument('dataset', choices=['train', 'test', 'review'],
                        help='name of the dataset to use')
    parser.add_argument('--output', '-o', type=str, default='output.txt',
                        help='filename of the result')
    parser.add_argument('--mode', '-m', choices=['train', 'infer'],
                        default='infer', help='mode of operation')
    parser.add_argument('--dry', '-d', action='store_true',
                        help='dry run only, the model is not saved')
    parser.add_argument('--reuse', '-r', action='store_true',
                        help='train upon existing model if exists')
    args = parser.parse_args()

    # fine tune the parameters
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = 's2s_{}'.format(timestamp)

    data = Video(args.folder, dtype=args.dataset)
