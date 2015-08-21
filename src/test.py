#!/usr/bin/env python
'''
| Filename    : test.py
| Description : A companion script to train.py for testing a trained model.
| Author      : Pushpendre Rastogi
| Created     : Thu Aug 20 23:28:02 2015 (-0400)
| Last-Updated: Fri Aug 21 00:04:27 2015 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 17
'''
import cPickle as pickle
import theano
import config
import numpy
import edge_dataset


def main(model_file):
    model = pickle.load(open(model_file, 'rb'))
    test_data = edge_dataset.load_data(
        start=0,
        stop=None,
        filename='res/bowman_wordnet_longer_shuffled_synset_relations.tsv',
        token_map='res/bowman_wordnet_longer_shuffled_synset_relations.map',
        first_column_has_y_label=True,
        first_column_of_map_file_has_index=True,
        return_composite_space_tuples=True,
    )
    (x_left, x_right, y_true) = test_data
    y_true = y_true.squeeze()
    y_hat = model.fprop((x_left, x_right)).argmax(axis=1).eval()
    assert y_hat.ndim == y_true.ndim
    print y_hat.shape, y_true.shape
    print "Test Accuracy: ", numpy.mean(y_hat == y_true)

if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(
        description='Test pickled pylearn model.')
    arg_parser.add_argument(
        '--model', default='tmp.tmp_best.pkl', type=str, help='Default={tmp.tmp_best.pkl}')
    arg_parser.add_argument(
        '--batch_size', default=1, type=int, help='Default={1}')
    config.args = arg_parser.parse_args()
    main(config.args.model)
