'''
| Filename    : edge_dataset.py
| Description : pylearn2 compatible dense_design_matrix objects that wrap Edge prediction data in csv files.
| Author      : Pushpendre Rastogi
| Created     : Tue Aug 18 00:45:47 2015 (-0400)
| Last-Updated: Mon Aug 24 17:27:23 2015 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 53
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
              return_composite_space_tuples=True,
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
    # First Shuffle.
    repeatable_shuffle(X, rng_for_shuffle_before_split)
    repeatable_shuffle(y, rng_for_shuffle_before_split)
    # Then
    X = np.asarray(X, dtype='int32')[start:, :]
    y = np.asarray(y, dtype='int32')
    y = y.reshape(y.shape[0], 1)[start:, :]
    if stop is not None:
        X = X[:stop - start, :]
        y = y[:stop - start, :]

    X_labels = int(X.max()) + 1
    y_labels = int(y.max()) + 1
    if return_composite_space_tuples:
        ddm = (X[:, 0:1], X[:, 1:2], y)
    else:
        y = convert_to_one_hot(y, y_labels)
        ddm = DenseDesignMatrix(
            X=X, y=y, X_labels=X_labels, y_labels=y_labels)
    return ddm


from pylearn2.space import IndexSpace, CompositeSpace
from pylearn2.datasets.vector_spaces_dataset import VectorSpacesDataset


class BowmanWordnetDataset(object):
    vertex_count = 3217
    dtype = 'int32'
    input_components = (IndexSpace(dim=1, max_labels=vertex_count,
                                   dtype=dtype),
                        IndexSpace(dim=1, max_labels=vertex_count,
                                   dtype=dtype))
    input_source = ('left_input', 'right_input')
    input_space = CompositeSpace(components=input_components)
    chromaticity = 3
    target_component = IndexSpace(dim=1, max_labels=chromaticity,
                                  dtype=dtype)
    target_source = ('target',)
    data_specs = (
        CompositeSpace(
            components=(input_components[0], input_components[1], target_component)),
        (input_source[0], input_source[1], target_source[0]))

    def __init__(self):
        pass


def BWD_dataset(portion_to_return, train_val_test=(0.8, 0.1, 0.1)):
    assert all(e >= 0 for e in train_val_test)
    assert sum(train_val_test) == 1
    total = 36772
    train_percent = train_val_test[0]
    val_percent = sum(train_val_test[0:2])
    train_stop = int(36772 * train_percent)
    valid_stop = int(36772 * val_percent)
    if portion_to_return == 'train':
        start, stop = (0, train_stop)
    elif portion_to_return == 'valid':
        start, stop = (train_stop + 1, valid_stop)
    else:
        start, stop = (valid_stop + 1, total)
    return VectorSpacesDataset(
        data=load_data(
            start=start,
            stop=stop,
            filename='res/bowman_wordnet_longer_shuffled_synset_relations.tsv',
            token_map='res/bowman_wordnet_longer_shuffled_synset_relations.map',
            first_column_has_y_label=True,
            first_column_of_map_file_has_index=True,
            return_composite_space_tuples=True),
        data_specs=BowmanWordnetDataset.data_specs)

BWD_input_space = BowmanWordnetDataset.input_space
BWD_input_source = BowmanWordnetDataset.input_source
BWD_target_source = BowmanWordnetDataset.target_source

import unittest


class TestModule(unittest.TestCase):

    def test_load_data(self):
        d = load_data()
        self.assertEqual((d[0].shape[0]), 36772)

    def test_BWD_dataset(self):
        d_train = BWD_dataset('train', train_val_test=(0.8, 0.1, 0.1)).data
        d_valid = BWD_dataset('valid', train_val_test=(0.8, 0.1, 0.1)).data
        d_test = BWD_dataset('test', train_val_test=(0.8, 0.1, 0.1)).data

        def examples_as_sets(d):
            (col1, col2, col3) = d
            return set((a[0], b[0], c[0])
                       for a, b, c
                       in zip(col1, col2, col3))

        s_train = examples_as_sets(d_train)
        s_valid = examples_as_sets(d_valid)
        s_test = examples_as_sets(d_test)
        self.assertEqual(0, len(s_train.intersection(s_valid)))
        self.assertEqual(0, len(s_train.intersection(s_test)))
        self.assertEqual(0, len(s_test.intersection(s_valid)))


if __name__ == '__main__':
    unittest.main()
