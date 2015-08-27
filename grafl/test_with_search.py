#!/usr/bin/env python
'''
| Filename    : test_with_search.py
| Description : Simple brute force search while testing.
| Author      : Pushpendre Rastogi
| Created     : Wed Aug 26 19:23:26 2015 (-0400)
| Last-Updated: Thu Aug 27 03:23:26 2015 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 95
'''
from test import get_model_predictions_on_data
from test import calculate_accuracy
import numpy
from numpy import log
from dataset.edge_dataset import BWD_dataset, BWD_vertex_count
import sys


def entropy(dist_arr, axis=1):
    assert axis == 1, str(NotImplemented)
    return numpy.array([-sum(e * log(e) for e in dist)
                        for dist in dist_arr], dtype='float64')

from contextlib import contextmanager

flush_counter = 0


@contextmanager
def flush_regularly(interval=100):
    global flush_counter
    if flush_counter % interval == 0:
        sys.stdout.flush()
    yield


def find_support(a1, a2, e):
    return sum(1
               for (e1, e2) in zip(a1, a2)
               if e1 == e and e2 == e)


def create_y_hat_from_search(y_dist, x_left_arrinp, x_right_arrinp, model_func,
                             max_points_to_look_at=1000, strategy=1):
    ''' Based on a distribution over labels.
    Use some heuristic to come up with y_hat.
    Params
    ------
    y_dist_test:
    x_left     :
    x_right    :
    model_func :
    '''
    assert strategy in [1, 2]
    # (x_left_train, x_right_train, y_true_train) = BWD_dataset('train').data
    # num_data_point = y_true_train.shape[0]
    num_data_point = y_dist.shape[0]
    # y_dist_train = model_func((x_left_train, x_right_train))
    # Figure out those cases where our distribution was wrong.
    # y_dist_argmax = y_dist_train.argmax(axis=1)
    # bad_dist_idx = [idx
    #                 for idx in range(num_train)
    #                 if y_dist_argmax[idx] != y_true_train[idx]]
    # good_dist_idx = [idx
    #                  for idx in range(num_train)
    #                  if idx not in bad_dist_idx]
    # Now for all cases figure out what does the best 1 hop transitive path
    # assign as the label.
    # This is basically the case of search with no back.
    # I can predict this as the label.
    y_hat = numpy.empty((num_data_point, 1), dtype='int32')
    x_left_arr = numpy.empty((BWD_vertex_count - 2, 1), dtype='int32')
    x_right_arr = numpy.empty_like(x_left_arr)
    eligible_hop_idx = numpy.empty_like(x_left_arr)
    model_func_retval = numpy.empty(
        (BWD_vertex_count - 2, 3), dtype='float32')
    # TODO: Fix the stupid memory leak
    for idx in range(min(num_data_point, max_points_to_look_at)):
        print idx
        x_left = x_left_arrinp[idx, 0]
        x_right = x_right_arrinp[idx, 0]
        tmp_idx = 0
        for hop_idx in range(BWD_vertex_count):
            if hop_idx != x_left and hop_idx != x_right:
                eligible_hop_idx[tmp_idx, 0] = hop_idx
                tmp_idx += 1
        x_left_arr.fill(x_left)
        x_right_arr.fill(x_right)
        path_prob = model_func(
            (x_left_arr, eligible_hop_idx), retval=model_func_retval)
        path_prob_2 = model_func(
            (eligible_hop_idx, x_right_arr), retval=model_func_retval)

        if strategy == 1:
            path1_argmax = path_prob.argmax(axis=1)
            path2_argmax = path_prob_2.argmax(axis=1)
            support = numpy.array([find_support(path1_argmax, path2_argmax, e)
                                   for e in [0, 1, 2]])
            y_hat[idx] = support.argmax()
        elif strategy == 2:
            path_prob = path_prob * path_prob_2
            path_prob /= path_prob.sum(axis=1, keepdims=True)
            y_hat[idx] = numpy.unravel_index(
                path_prob.argmax(), (BWD_vertex_count - 2, 3))[1]
    return y_hat


def main(model_file, strategy):
    y_dist, (x_left, x_right, y_true), model_func = get_model_predictions_on_data(
        model_file, datatype='train')
    # Now do not look at y_true at all.
    mptla = 1000
    y_hat = create_y_hat_from_search(y_dist, x_left, x_right, model_func,
                                     max_points_to_look_at=mptla,
                                     strategy=strategy)
    _ = calculate_accuracy(y_hat[:mptla],
                           y_true[:mptla])

if __name__ == '__main__':
    main(
        r"output/res/experiments/BWD-projection-identity_sub_glue-Softmax.pkl",
        1)
