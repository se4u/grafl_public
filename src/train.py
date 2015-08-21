#!/usr/bin/env python
'''
| Filename    : cross_validate.py
| Description : Invoke this script from command line to cross-validate the performance of a model.
| Author      : Pushpendre Rastogi
| Created     : Mon Aug 17 23:36:23 2015 (-0400)
| Last-Updated: Fri Aug 21 00:08:08 2015 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 34
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


def get_best_model_save_path(save_path):
    return save_path + '_best.pkl'


def test():
    import test as test_module
    test_module.main(get_best_model_save_path(config.args.save_path))


def train():
    save_path = config.args.save_path
    config_yaml = (open(config.args.yaml).read()) % dict(
        save_path=save_path,
        best_model_save_path=get_best_model_save_path(save_path))

    if config.args.debug_load:
        trainer = pdb.runcall(yaml_parse.load, config_yaml)
    else:
        trainer = yaml_parse.load(config_yaml)

    if config.args.debug_main_loop:
        pdb.runcall(trainer.main_loop)
    else:
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
    arg_parser.add_argument(
        '--debug_load', default=0, type=int, help='Default={0}')
    arg_parser.add_argument(
        '--debug_main_loop', default=0, type=int, help='Default={0}')
    config.args = arg_parser.parse_args()
    random.seed(config.args.seed)
    np.random.seed(config.args.seed)
    try:
        main()
    except:
        _, __, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
