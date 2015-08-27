'''
| Filename    : multilinear_mlp.py
| Description : An MLP Layer with tensor activations.
| Author      : Pushpendre Rastogi
| Created     : Fri Aug 21 20:23:06 2015 (-0400)
| Last-Updated: Sat Aug 22 19:12:53 2015 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 30
'''
import theano.tensor
import numpy
from functools import wraps
from pylearn2.compat import OrderedDict
from pylearn2.space import VectorSpace, CompositeSpace
from pylearn2.models.mlp import Layer
from pylearn2.utils import sharedX


class MultiLinearTransform(object):

    """
    A generic class describing a MultiLinearTransform.
    This class mirrors LinearTransform in pylearn2.
    """

    def get_params(self):
        raise NotImplementedError()

    def get_weights_topo(self):
        raise NotImplementedError()

    def set_batch_size(self, batch_size):
        raise NotImplementedError()


class ThreeDTensorMul(MultiLinearTransform):

    def __init__(self, T, is_symbolic=True):
        self._T = T
        self.tensordot_f = (theano.tensor.tensordot
                            if is_symbolic
                            else numpy.tensordot)

    @wraps(MultiLinearTransform.get_params)
    def get_params(self):
        return [self._T]

    @staticmethod
    def tensor_matrix_elemwise_per_slice(T, M):
        ''' Multiply each slice of an a x b x c tensor
        with an a x b matrix without for loops in python.

        The method to do is to add a broadcast to matrix M.
        (a, b, c) = T.shape
        assert M.shape == (a, b)
        '''
        return (T * numpy.expand_dims(M, axis=2)
                if isinstance(M, numpy.ndarray)
                else T * M.dimshuffle(0, 1, 'x'))

    def bilinear_product(self, a, b):
        ''' Return the bilinear product of a 3D tensor [d1 x d2 x d3]
        with 'a' and 'b' which are [J x d1], [J x d2] matrices, respectively.
        '''
        intermediate = self.tensordot_f(a, self._T, axes=[1, 0])
        activation = self.tensor_matrix_elemwise_per_slice(
            intermediate, b).sum(axis=1)
        return activation


class MultiLinear(Layer):

    """
    A model which receives a 2-factor composite space as input and
    computes the output activation as a tensor product of
    the inputs. The output is:

    output = [e1^T M1 e2, e1^T M2 e2, ...]

    There is no bias, no multiplier.

    TODO: Figure out how its output can be added to the output of a linear layer
    and then nonlinearly transformed?
    """

    def __init__(self, dim, layer_name,
                 irange=0.01, **kwargs):
        assert irange is not None
        super(MultiLinear, self).__init__(**kwargs)
        self.dim = dim
        self.layer_name = layer_name
        self.irange = irange

    @wraps(Layer.get_lr_scalers)
    def get_lr_scalers(self):
        'Essentially we are bypassing it.'
        return OrderedDict()

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        self.input_space = space
        assert isinstance(space, CompositeSpace)
        component_dims = [component.get_total_dimension()
                          for component in space.components]
        assert len(component_dims) == 2
        #self.input_dim = reduce(lambda x, y: x * y, component_dims, 1)
        input_dim = tuple(component_dims)
        self.input_dim = input_dim
        self.output_space = VectorSpace(self.dim)
        rng = self.mlp.rng
        T = rng.uniform(-self.irange, self.irange,
                        (input_dim[0], input_dim[1], self.dim))
        T = sharedX(T)
        T.name = self.layer_name + '_T'
        self.transformer = ThreeDTensorMul(T)

    @wraps(Layer._modify_updates)
    def _modify_updates(self, updates):
        pass

    @wraps(Layer.get_params)
    def get_params(self):
        T, = self.transformer.get_params()
        assert T.name is not None
        return [T]

    @staticmethod
    def sanitize_coeff(coeff):
        coeff = (float(coeff)
                 if isinstance(coeff, str)
                 else coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        return coeff

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeff):
        coeff = self.sanitize_coeff(coeff)
        T, = self.transformer.get_params()
        return coeff * theano.tensor.sqr(T).sum()

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeff):
        coeff = self.sanitize_coeff(coeff)
        T, = self.transformer.get_params()
        return coeff * abs(T).sum()

    @wraps(Layer.get_weights)
    def get_weights(self):
        T, = self.transformer.get_params()
        return T.get_value()

    @wraps(Layer.set_weights)
    def set_weights(self, weights):
        T, = self.transformer.get_params()
        T.set_value(weights)

    @wraps(Layer.set_biases)
    def set_biases(self, biases):
        raise NotImplementedError()

    @wraps(Layer.get_biases)
    def get_biases(self):
        raise NotImplementedError()

    @wraps(Layer.get_weights_format)
    def get_weights_format(self):
        return ('v', 'v', 'h')

    @wraps(Layer.get_weights_topo)
    def get_weights_topo(self):
        raise NotImplementedError()

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):
        T, = self.transformer.get_params()
        assert T.ndim == 3
        # sq_T = theano.tensor.sqr(T)
        # Prepare an orderedDict with values to monitor.
        return OrderedDict()

    def _multi_linear_part(self, state_below):
        self.input_space.validate(state_below)
        p = self.transformer.bilinear_product(
            state_below[0], state_below[1])
        if self.layer_name is not None:
            p.name = self.layer_name + '_z'
        return p
        # Do not format state_below.

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        p = self._multi_linear_part(state_below)
        return p

    @wraps(Layer.cost)
    def cost(self, T, Y_hat):
        raise NotImplementedError()

    @wraps(Layer.cost_from_cost_matrix)
    def cost_from_cost_matrix(self, cost_matrix):
        raise NotImplementedError()

    @wraps(Layer.cost_matrix)
    def cost_matrix(self, Y, Y_hat):
        raise NotImplementedError()


class MultiLinearRectified(MultiLinear):

    def __init__(self, left_slope=0.0, **kwargs):
        super(MultiLinearRectified, self).__init__(**kwargs)
        self.left_slope = left_slope

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        p = self._multi_linear_part(state_below)
        return theano.tensor.switch(p > 0., p, self.left_slope * p)

    @wraps(Layer.cost)
    def cost(self, *args, **kwargs):
        raise NotImplementedError()


import unittest


class TestThreeDTensorMul(unittest.TestCase):

    def test_tensor_matrix_elemewise_per_slice(self):
        T = numpy.dstack(
            (numpy.identity(2), numpy.identity(2)))
        M = 2 * numpy.ones((2, 2))
        p = ThreeDTensorMul(None).tensor_matrix_elemwise_per_slice(
            T, M)
        numpy.testing.assert_array_equal(p, 2 * T)

    def test_bilinear_product(self):
        num_row = 10
        num_slice = 4
        T = numpy.random.rand(2, 3, num_slice)
        a = numpy.random.rand(num_row, 2)
        b = numpy.random.rand(num_row, 3)
        returned_p = ThreeDTensorMul(T, is_symbolic=False).bilinear_product(
            a, b)
        expected_p = numpy.empty((num_row, num_slice), dtype='float64')
        for ret_col_idx in range(num_slice):
            for ret_row_idx in range(num_row):
                T_slice = T[:, :, ret_col_idx]
                a_row = a[ret_row_idx]
                b_row = b[ret_row_idx]
                expected_p[ret_row_idx, ret_col_idx] = numpy.dot(
                    numpy.dot(a_row, T_slice), b_row)
        numpy.testing.assert_array_almost_equal(returned_p, expected_p)
        numpy.testing.assert_array_equal(returned_p, expected_p)


if __name__ == '__main__':
    unittest.main()
