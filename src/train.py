#!/usr/bin/env python
'''
| Filename    : cross_validate.py
| Description : Invoke this script from command line to cross-validate the performance of a model.
| Author      : Pushpendre Rastogi
| Created     : Mon Aug 17 23:36:23 2015 (-0400)
| Last-Updated: Thu Aug 20 00:46:36 2015 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 29
'''
import config
from config import open
import sklearn
import numpy as np
import random
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import pdb
import traceback
import sys


def get_acc(model, X, y):
    predicted_output = model.get_output_for(X).eval().argmax(axis=1)
    actual_output = y.argmax(axis=1)
    fold_acc = (predicted_output == actual_output).mean()
    return fold_acc


def test():
    import cPickle as pickle
    import lib_score_builder
    models = pickle.load(open(config.args.save_path))
    from edge_dataset import load_data
    data = load_data(filename='res/hypernymOf_partOf.default.input.tsv',
                     token_map='res/hypernymOf_partOf.default.input.map')
    X = data.X
    y = data.y
    accuracy = ([get_acc(m, X, y) for m in models]
                if isinstance(models, list)
                else [get_acc(models, X, y)])
    print "Final Accuracy", float(sum(accuracy)) / len(accuracy)


def train():
    config_yaml = (open(config.args.yaml).read()) % dict(
        save_path=config.args.save_path)
    trainer = yaml_parse.load(config_yaml)
    trainer.main_loop()


def main():
    if not config.args.skip_train:
        train()
    if config.args.do_test:
        test()


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(description='Cross Validate')
    arg_parser.add_argument('--seed', default=0, type=int, help='Default={0}')
    arg_parser.add_argument('--fold', default=5, type=int, help='Default={5}')
    arg_parser.add_argument(
        '--test_percentage', default=10, type=float, help='Default={10}')
    arg_parser.add_argument(
        '--train_percentage', default=90, type=float, help='Default={90}')
    arg_parser.add_argument(
        '--architecture', default='nn', type=str, help='Default={nn}')
    arg_parser.add_argument('--input', default='toy_synset_relations.tsv',
                            type=str, help='Default={toy_synset_relations.tsv}')
    arg_parser.add_argument(
        '--yaml', default='train.yaml', type=str, help='Default={toy.yaml}')
    arg_parser.add_argument(
        '--save_path', default='tmp.tmp', type=str, help='Default={tmp.tmp}')
    arg_parser.add_argument(
        '--do_test', default=1, type=int, help='Default={1}')
    arg_parser.add_argument(
        '--skip_train', default=0, type=int, help='Default={1}')
    config.args = arg_parser.parse_args()
    random.seed(config.args.seed)
    np.random.seed(config.args.seed)
    try:
        main()
    except:
        _, __, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
