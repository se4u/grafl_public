'''
| Filename    : cross_validate.py
| Description : Invoke this script from command line to cross-validate the performance of a model.
| Author      : Pushpendre Rastogi
| Created     : Mon Aug 17 23:36:23 2015 (-0400)
| Last-Updated: Tue Aug 18 16:18:50 2015 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 9
'''
import config
from config import open
import sklearn
import numpy as np
import random
from pylearn2.utils import serial


def main():
    config_yaml = open("toy.yaml").read()
    trainer = serial.load_train_file(
        config_yaml % dict(save_path='tmp.tmp'))
    trainer.main_loop()

if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(description='Cross Validate')
    arg_parser.add_argument('--seed', default=0, type=int, help='Default={0}')
    arg_parser.add_argument('--fold', default=5, type=int, help='Default={5}')
    arg_parser.add_argument('--test_percentage', default=10, type=float, help='Default={10}')
    arg_parser.add_argument('--train_percentage', default=90, type=float, help='Default={90}')
    arg_parser.add_argument('--architecture', default='nn', type=str, help='Default={nn}')
    arg_parser.add_argument('--input', default='toy_synset_relations.tsv', type=str, help='Default={toy_synset_relations.tsv}')
    config.args = arg_parser.parse_args()
    random.seed(config.args.seed)
    np.random.seed(config.args.seed)
    main()
