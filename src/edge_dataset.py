'''
| Filename    : edge_dataset.py
| Description : pylearn2 compatible dense_design_matrix objects that wrap Edge prediction data in csv files.
| Author      : Pushpendre Rastogi
| Created     : Tue Aug 18 00:45:47 2015 (-0400)
| Last-Updated: Tue Aug 18 19:45:19 2015 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 9
'''
import csv
import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from config import open

def load_data(start=0, stop=None, filename='', token_map='', header = False):
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        token_map = dict((t, int(idx))
                         for (t, idx)
                         in [e.strip().split()
                             for e
                             in open(token_map)])
        X = []
        y = []
        for row in reader:
            # Skip the first row containing the string names of each attribute
            if header:
                header = False
                continue
            # Convert the row into numbers
            row = [float(token_map[elem]) for elem in row]
            X.append(row[:-1])
            y.append(row[-1])
    X = np.asarray(X)[start:, :]
    y = np.asarray(y)
    y = y.reshape(y.shape[0], 1)[start:, :]
    if stop is not None:
        X = X[:stop-start, :]
        y = y[:stop-start, :]
    return DenseDesignMatrix(X=X, y=y)
