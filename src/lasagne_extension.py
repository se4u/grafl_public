'''
| Filename    : lasagne_extension.py
| Description : Lasagne extensions for the neural tensor network activation.
| Author      : Pushpendre Rastogi
| Created     : Mon Aug 17 01:57:13 2015 (-0400)
| Last-Updated: Mon Aug 17 02:04:11 2015 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 2
'''
import lasagne
import unittest


class GlorotBilinearForm(lasagne.init.Initializer):

    """
    GlorotBilinearForm(initializer=lasagne.init.Uniform, gain=1.0,
                       symmetry=None)

    A Bilinear form is a function that maps two vectors to a real number
    as follows f(x, y) = x^T M y. M is a simple matrix and x^T M y is computed
    through simple matrix multiplication.

    A neural layer can be computed by concatenating `k` bilinear forms.
    y = f([a^T M_1 b; a^T M_2 b; ...; a^T M_k b])

    Parameters
    ----------
    initializer : A :class:`lasagne.init.Initializer` instance which can be
        initialized by specifying a standard deviation `std` parameter.

    gain : Refer to :class:`lasagne.init.Glorot` for details.

    symmetry : The matrices M_I can be forced to be symmetric or antisymmetric
       or left as it is after sampling. The acceptable values for this
       parameter are {'symmetric', 'antisymmetric', None}

    """

    def __init__(self, initializer=lasagne.init.Uniform, gain=1.0, symmetry=None):
        self.gain = (np.sqrt(2) if gain == 'relu' else 1.0) * gain
        self.initializer = initializer
        assert symmetry in [None, 'symmetric', 'antisymmetric']
        self.symmetry = symmetry

    def sample(self, shape):
        assert len(shape) < 3, \
            "This initializer only works with shapes of length 3 or more"
        dim_a, dim_b, k = shape[:3]
        std = self.gain * np.sqrt(2.0 / (dim_a + dim_b + k))
        M = self.initializer(std=std).sample(shape)
        if self.symmetry == 'symmetric':
            M += numpy.swapaxes(M, 0, 1)
            M /= 2
        elif self.symmetric == 'antisymmetric':
            M -= numpy.swapaxes(M, 0, 1)
            M /= 2
        else:
            pass
        return M


class NeuralTensorNetworkLayer(lasagne.layers.Layer):

    """
    NeuralTensorNetworkLayer(self, incoming, num_units, bifurcation_point,
        W=init.GlorotUniform(),
        b=init.Constant(0.), T=GlorotBilinearForm(),
        nonlinearity=nonlinearities.rectify, **kwargs)

    A NeuralTensorNetworkLayer adds an extra 3D tensor component to the usual
    neural network activations. It is most used for representing a potential
    or score function given two input vector a and b. The output is computed
    as `f( a T b + W [a; b] + bias)` where f is the nonlinearity.

    Parameters
    ----------
    incoming : a concatenated :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    num_units : int
        The number of output units of the layer

    bifurcation_point : int
        The dimensionality of the first part of the input.

    W : Theano shared variable, numpy array, callable or None
        An initializer for the weights W. The matrix part of
        the activations.

    b : Theano shared variable, numpy array, callable or None
        An initializer for the weights b. The bias part of the
        activations.

    T : Theano shared variable, numpy array, callable or None
        An initializer for the weights T. The bilinear/tensor
        part of the activations. It has a `symmetry` property
        which has one of three values {'symmetric', 'antisymmetric', None}
        See :class:`GlorotBilinearForm` for details.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations.
        If None is provided, then the layer will be linear.

    References
    ----------
    .. [1] Socher, Richard and Chen, Danqi and Manning, Christopher D. and
           Ng, Andrew: Reasoning with neural tensor networks for knowledge base
           completion. NIPS (2013)
    """

    def __init__(self, incoming, num_units, bifurcation_point,
                 W=init.GlorotUniform(),
                 b=init.Constant(0.), T=GlorotBilinearForm(symmetry=None),
                 nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(NeuralTensorNetworkLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)
        self.num_units = num_units
        self.bifurcation_point = bifurcation_point
        self.symmetry = T.symmetry
        if len(self.input_shape) >= 2:
            raise RuntimeError(
                "I did not foresee %d dimensional input." % len(self.input_shape))
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        self.b = (None
                  if b is None
                  else self.add_param(b, (num_units,), name="b",
                                      regularizable=False))
        self.T = self.add_param(
            T, (num_inputs, num_inputs, num_units), name="T")

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        bp = self.bifurcation_point
        activation = T.tensordot(
            T.tensordot(self.T, input[bp:], axes=0),
            input[:bp], axes=1)

        if self.W is not None:
            activation += T.dot(input, self.W)
        if self.b is not None:
            activation += self.b.dimshuffle('x', 0)

        return self.nonlinearity(activation)


class TestNeuralTensorNetworkLayer(unittest.TestCase):

    def test_get_output_for(self):
        pass


class TestGlorotBilinearForm(unittest.TestCase):

    def test_sample(self):
        pass


if __name__ == '__main__':
    unittest.main()
