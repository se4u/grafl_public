'''
| Filename    : edge_dataset.py
| Description : pylearn2 compatible dense_design_matrix objects that wrap Edge prediction data in csv files.
| Author      : Pushpendre Rastogi
| Created     : Tue Aug 18 00:45:47 2015 (-0400)
| Last-Updated: Fri Aug 21 18:12:55 2015 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 39
'''
import csv
import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from config import open


def convert_to_one_hot(y, y_labels):
    arr = np.zeros((len(y), y_labels), dtype='int32')
    for idx in range(len(y)):
        arr[idx, y[idx]] = 1
    return arr


def repeatable_shuffle(x, seed):
    state = np.random.get_state()
    np.random.seed(seed)
    np.random.shuffle(x)
    np.random.set_state(state)
    return


def load_data(start=0, stop=None,
              filename='res/bowman_wordnet_longer_shuffled_synset_relations.tsv',
              token_map='res/bowman_wordnet_longer_shuffled_synset_relations.map',
              header=False,
              first_column_has_y_label=True,
              first_column_of_map_file_has_index=True,
              return_composite_space_tuples=False,
              split_percentage_for_training=100,
              portion_to_return='train',
              rng_for_shuffle_before_split=1234):
    token_map = dict(((c1, int(c0)) if first_column_of_map_file_has_index else (c0, int(c1)))
                     for (c0, c1)
                     in [e.strip().split()
                         for e
                         in open(token_map)])

    with open(filename, 'r') as f:

        X = []
        y = []
        for row in f:
            if header:
                header = False
                continue
            # Convert the row into numbers
            row = [float(token_map[elem]) for elem in row.strip().split()]
            if first_column_has_y_label:
                X.append(row[1:])
                y.append(row[0])
            else:
                X.append(row[:-1])
                y.append(row[-1])
    X = np.asarray(X, dtype='int32')[start:, :]
    y = np.asarray(y, dtype='int32')
    y = y.reshape(y.shape[0], 1)[start:, :]
    if stop is not None:
        X = X[:stop - start, :]
        y = y[:stop - start, :]

    X_labels = int(X.max()) + 1
    y_labels = int(y.max()) + 1
    import numpy
    repeatable_shuffle(X, rng_for_shuffle_before_split)
    repeatable_shuffle(y, rng_for_shuffle_before_split)
    idx_for_split = int(
        float(split_percentage_for_training) / 100 * X.shape[0])
    import pdb
    # pdb.set_trace()
    if portion_to_return == 'test':
        X = X[idx_for_split:]
        y = y[idx_for_split:]
    elif portion_to_return == 'train':
        X = X[:idx_for_split]
        y = y[:idx_for_split]
    else:
        pass
    if return_composite_space_tuples:
        ddm = (X[:, 0:1], X[:, 1:2], y)
    else:
        y = convert_to_one_hot(y, y_labels)
        ddm = DenseDesignMatrix(
            X=X, y=y, X_labels=X_labels, y_labels=y_labels)
    return ddm

if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(
        description='Edge Dataset Loading Script')
    arg_parser.add_argument('--split_percentage_for_training', default=80,
                            type=int, help='Default={80}')
    args = arg_parser.parse_args()
    load_data(split_percentage_for_training=args.split_percentage_for_training)
