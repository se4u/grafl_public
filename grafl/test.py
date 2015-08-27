#!/usr/bin/env python
'''
| Filename    : test.py
| Description : A companion script to train.py for testing a trained model.
| Author      : Pushpendre Rastogi
| Created     : Thu Aug 20 23:28:02 2015 (-0400)
| Last-Updated: Thu Aug 27 02:10:42 2015 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 69
'''
import cPickle as pickle
import numpy
import theano
import config
from dataset.edge_dataset import BWD_dataset


def make_model_func(model, batch_size=1000):
    a = numpy.empty((batch_size, 1), dtype='int32')
    b = numpy.empty((batch_size, 1), dtype='int32')
    th_a = theano.shared(a)
    th_b = theano.shared(b)
    ff = theano.function((), model.fprop((th_a, th_b)))

    def model_func(x, retval=None, borrow=True):
        (x_left, x_right) = x
        num_point = x_left.shape[0]
        if retval is None:
            retval = numpy.empty((num_point, 3), dtype='float32')
        for idx in range(0, num_point, batch_size):
            th_a.set_value(x_left[idx:idx + batch_size], borrow=borrow)
            th_b.set_value(x_right[idx:idx + batch_size], borrow=borrow)
            # TODO: Investigate why numpy.copyto gives wrong result?
            # numpy.copyto(a, x_left[idx:idx + batch_size])
            # numpy.copyto(b, x_right[idx:idx + batch_size])
            retval[idx:idx + batch_size] = ff()
        return retval

    return model_func


def get_model_predictions_on_test_data(model_file, batch_size=1000):
    model = pickle.load(open(model_file, 'rb'))
    test_data = BWD_dataset('test').data
    (x_left, x_right, y_true) = test_data
    model_func = make_model_func(model, batch_size=batch_size)
    y_dist = model_func((x_left, x_right))
    y_dist_gold = model.fprop((x_left, x_right)).eval()
    numpy.testing.assert_array_equal(y_dist, y_dist_gold)
    print "Gold test passed"
    return y_dist, (x_left, x_right, y_true), model_func


def calculate_accuracy(y_hat, y_true):
    y_true = y_true.squeeze()
    y_hat = y_hat.squeeze()
    assert y_hat.ndim == y_true.ndim
    print y_hat.shape, y_true.shape
    accuracy = numpy.mean(y_hat == y_true)
    print "Test Accuracy: ", accuracy
    return accuracy


def main(model_file):
    import pdb
    import traceback
    import sys
    try:
        y_dist, (_, __, y_true), ___ = get_model_predictions_on_test_data(
            model_file)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
    y_hat = y_dist.argmax(axis=1)
    _ = calculate_accuracy(y_hat, y_true)

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
