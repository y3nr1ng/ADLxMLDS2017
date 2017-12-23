import argparse
import pandas as pd
import re

def parse_arguments():
    parser = argparse.ArgumentParser('Preprocessor')
    parser.add_argument('tags', type=str, help='raw tags')
    parser.add_argument('rules', type=str, help='tags to keep')
    parser.add_argument('-o', '--output', type=str, default='result.csv')
    return parser.parse_args()

def load_tags(path):
    df = pd.read_csv(path, index_col=0, names=['id', 'tags'])
    return df

def load_rules(path):
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    return set(lines)

def process(tags, rules):
    pattern = re.compile('([^\t]+):[\d]*')
    loc = tags.columns.get_loc('tags')
    for i, row in tags.iterrows():
        t = pattern.findall(row['tags'])
        t = set(t)
        t = list(t.intersection(rules))

        tags.iat[i, loc] = '\t'.join(t)

if __name__ == '__main__':
    args = parse_arguments()

    tags = load_tags(args.tags)
    rules = load_rules(args.rules)

    process(tags, rules)
    n_images = len(tags)
    print('{} images processed'.format(n_images))

    tags = tags[tags['tags'] != '']
    print('dropped {} images'.format(n_images-len(tags)))

    tags.to_csv(args.output, header=False)
