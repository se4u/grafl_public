#!/usr/bin/env python
'''
| Filename    : test.py
| Description : A companion script to train.py for testing a trained model.
| Author      : Pushpendre Rastogi
| Created     : Thu Aug 20 23:28:02 2015 (-0400)
| Last-Updated: Tue Aug 25 01:41:40 2015 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 22
'''
import cPickle as pickle
import theano
import config
import numpy
from edge_dataset import BWD_dataset


def main(model_file):
    model = pickle.load(open(model_file, 'rb'))
    test_data = BWD_dataset('test').data
    import pdb
    # pdb.set_trace()
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
