'''
| Filename    : glue_composite_layer.py
| Description : Classes/functions to transform CompositeLayer's output to VectorSpace.
| Author      : Pushpendre Rastogi
| Created     : Sat Aug 22 18:38:07 2015 (-0400)
| Last-Updated: Wed Aug 26 20:14:16 2015 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 14
'''
from pylearn2.models.mlp import Layer
from functools import wraps
from pylearn2.space import VectorSpace
import theano.tensor


class GlueLayer(Layer):

    def __init__(self, dim, layer_name, operation, nonlinearity=None, **kwargs):
        super(GlueLayer, self).__init__(**kwargs)
        self.dim = dim
        self.layer_name = layer_name
        self.nonlinearity = nonlinearity
        self.operation = operation
        self.output_space = VectorSpace(self.dim)
        self._params = []

    @wraps(Layer.set_input_space)
    def set_input_space(self, input_space):
        self.input_space = input_space

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        p = reduce(self.operation, state_below)
        if self.nonlinearity is not None:
            p = self.nonlinearity(p)
        p.name = self.layer_name + '_z'
        return p

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeffs):
        return 0.0


class Rectifier(object):

    def __init__(self, left_slope=0.0):
        self.left_slope = left_slope

    def __call__(self, p):
        return theano.tensor.switch(p > 0., p, self.left_slope * p)
