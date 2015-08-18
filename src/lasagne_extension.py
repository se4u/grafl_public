'''
| Filename    : lasagne_extension.py
| Description : Lasagne extensions for the neural tensor network activation.
| Author      : Pushpendre Rastogi
| Created     : Mon Aug 17 01:57:13 2015 (-0400)
| Last-Updated: Mon Aug 17 21:19:56 2015 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 64
'''
import lasagne
import unittest
import numpy as np
import theano.tensor


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
    SYMMETRIC = 'sym'
    ANTISYMMETRIC = 'antisym'
    SYMMETRY_FLAGS = [None, SYMMETRIC, ANTISYMMETRIC]

    def __init__(self, initializer=lasagne.init.Uniform, gain=1.0, symmetry=None):
        self.gain = (np.sqrt(2) if gain == 'relu' else gain)
        self.initializer = initializer
        assert symmetry in self.SYMMETRY_FLAGS
        self.symmetry = symmetry

    def sample(self, shape):
        assert len(shape) >= 3, \
            "This initializer only works with shapes of length 3 or more"
        if (self.symmetry in [self.SYMMETRIC, self.ANTISYMMETRIC]
                and shape[0] != shape[1]):
            raise ValueError(
                "Symmetry can only be enforced if all order 3 slices are square")

        dim_a, dim_b, k = shape[:3]
        std = self.gain * np.sqrt(2.0 / (dim_a + dim_b + k))
        M = self.initializer(std=std).sample(shape)
        if self.symmetry == self.SYMMETRIC:
            M += np.swapaxes(M, 0, 1)
            M /= 2
        elif self.symmetry == self.ANTISYMMETRIC:
            M -= np.swapaxes(M, 0, 1)
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
                 W=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Constant(0.),
                 T=GlorotBilinearForm(symmetry=None),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 use_numpy=False,
                 **kwargs):
        super(NeuralTensorNetworkLayer, self).__init__(incoming, **kwargs)
        process_fnc = (lambda x: x.eval()
                       if use_numpy
                       else lambda x: x)
        self.nonlinearity = (lasagne.nonlinearities.identity
                             if nonlinearity is None
                             else nonlinearity)
        self.num_units = num_units
        self.bifurcation_point = bifurcation_point
        if len(self.input_shape) > 2:
            raise ValueError("I don't know what do with >2 dimensional input")
        num_inputs = int(np.prod(self.input_shape[1:]))
        self.W = (None
                  if W is None
                  else process_fnc(self.add_param(
                      W, (num_inputs, num_units), name="W")))
        self.b = (None
                  if b is None
                  else process_fnc(self.add_param(
                      b, (num_units,), name="b", regularizable=False)))
        T_shape = (
            bifurcation_point, num_inputs - bifurcation_point, num_units)
        self.T = process_fnc(self.add_param(T, T_shape, name="T"))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    @staticmethod
    def tensor_matrix_elemwise_per_slice(T, M):
        ''' Multiply each slice of an a x b x c tensor
        with an a x b matrix without for loops in python.

        The method to do is to add a broadcast to matrix M.
        (a, b, c) = T.shape
        assert M.shape == (a, b)
        '''
        if isinstance(M, np.ndarray):
            return T * np.expand_dims(M, axis=2)
        elif isinstance(M, theano.tensor.gof.Variable):
            return T * M.dimshuffle(0, 1, 'x')
        else:
            raise RuntimeError(str(type(T)))

    def get_output_for(self, input_var, use_numpy=False, **kwargs):
        tensor_module = (np
                         if use_numpy
                         else theano.tensor)

        if input_var.ndim > 2:
            # if the input_var has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input_var = input_var.flatten(2)

        bp = self.bifurcation_point
        intermediate = tensor_module.tensordot(
            input_var[:, :bp], self.T, axes=[1, 0])
        activation = self.tensor_matrix_elemwise_per_slice(
            intermediate, input_var[:, bp:]).sum(axis=1)

        if self.W is not None:
            activation += tensor_module.dot(input_var, self.W)
        if self.b is not None:
            activation += self.b.dimshuffle('x', 0)

        return self.nonlinearity(activation)


class TestNeuralTensorNetworkLayer(unittest.TestCase):

    def get_output_for_symmetric_impl(self, use_numpy):
        num_units = 2
        T = np.array([[[1, 1], [2, 2], [1, 1]],
                      [[2, 2], [0, 0], [3, 3]],
                      [[1, 1], [3, 3], [1, 1]]])
        obj = NeuralTensorNetworkLayer(
            incoming=(None, 3 + 3),
            num_units=num_units,
            bifurcation_point=3,
            W=None,
            b=None,
            T=(GlorotBilinearForm(
                symmetry=GlorotBilinearForm.SYMMETRIC)
               if use_numpy
               else T),
            nonlinearity=None,
            use_numpy=use_numpy)
        if use_numpy:
            self.assertEqual(obj.num_units, num_units)
            self.assertEqual(obj.T.shape, (3, 3, num_units))
            self.assertEqual(np.sum(obj.T[:, :, 0] - obj.T[:, :, 0].T), 0.0)
            self.assertEqual(np.sum(obj.T[:, :, 1] - obj.T[:, :, 1].T), 0.0)
        #------------------------------------------------------------#
        # Test output when the tensor is Symmetric and vec_a = vec_b #
        #------------------------------------------------------------#
        # np.expand_dims(T, axis=2).repeat(repeats=num_units, axis=2)
        obj.T = T
        input_var = np.array([[1,  2,  3, 0, 1, 2],
                              [-1, -2, -3, -0, -1, -2]],
                             dtype=np.int32)
        output = obj.get_output_for(input_var, use_numpy=use_numpy)
        if not use_numpy:
            output = output.eval()
        batch_size = input_var.shape[0]
        self.assertEqual(output.shape, (batch_size, num_units))
        for idx in range(input_var.shape[0]):
            for idx_num_unit in range(1, num_units):
                self.assertEqual(output[idx][0], output[idx][idx_num_unit])
        self.assertEqual(
            output[0, 0], np.dot(np.dot(T[:, :, 0], [1, 2, 3]), [0, 1, 2]))
        self.assertEqual(output[1, 0], output[0, 0])
        #------------------------------------------------------------------#
        # Test output=0 when vec_a = vec_b and the tensor is AntiSymmetric #
        #------------------------------------------------------------------#
        T = np.array([[0, -2, -1],
                      [2, 0, -3],
                      [1, 3, 0]])
        obj.T = np.expand_dims(T, axis=2).repeat(repeats=num_units, axis=2)
        input_var = np.array([[1,  2,  3, 1, 2, 3]],
                             dtype=np.int32)
        output = obj.get_output_for(input_var, use_numpy=True)
        self.assertEqual(output[0, 0], 0)

    def test_get_output_for_symmetric(self):
        self.get_output_for_symmetric_impl(False)
        self.get_output_for_symmetric_impl(True)

    def test_get_output_for_random(self):
        num_units = 1
        obj = NeuralTensorNetworkLayer(
            incoming=(None, 3 + 2),
            num_units=num_units,
            bifurcation_point=3,
            W=None,
            b=None,
            T=GlorotBilinearForm(symmetry=None),
            nonlinearity=None,
            use_numpy=True)
        self.assertEqual(obj.T.shape, (3, 2, num_units))
        T = np.array([[[1], [2], ],
                      [[2], [0], ],
                      [[1], [3], ]], dtype=np.int32)
        input_var = np.array([[1,  2,  3, 2, 1]],
                             dtype=np.int32)
        obj.T = T
        output = obj.get_output_for(input_var, use_numpy=True)
        self.assertEqual(output[0, 0], 27)


class TestGlorotBilinearForm(unittest.TestCase):

    def setUp(self):
        from mock import MagicMock
        self.initializer = MagicMock()

    def assert_almostcalled_with(self, mock_obj, *args, **kwargs):
        mock_args, mock_kwargs = mock_obj.call_args
        for idx in range(len(args)):
            self.assertAlmostEqual(args[idx], mock_args[idx])
        for key in kwargs:
            self.assertAlmostEqual(kwargs[key], mock_kwargs[key])

    def test_sample(self):
        #--------------------------------------------------#
        # Test that the standard deviations are set right. #
        #--------------------------------------------------#
        obj = GlorotBilinearForm(initializer=self.initializer,
                                 gain=1.0,
                                 symmetry=GlorotBilinearForm.SYMMETRIC)
        sample = obj.sample((1, 1, 1))
        self.initializer.assert_called_with(std=np.sqrt(2.0 / 3.0))

        obj = GlorotBilinearForm(initializer=self.initializer,
                                 gain=1.0,
                                 symmetry=None)
        sample = obj.sample((1, 2, 3))
        self.initializer.assert_called_with(std=np.sqrt(2.0 / 6.0))
        obj = GlorotBilinearForm(initializer=self.initializer,
                                 gain='relu',
                                 symmetry=GlorotBilinearForm.SYMMETRIC)
        sample = obj.sample((1, 1, 1))
        self.assert_almostcalled_with(self.initializer, std=np.sqrt(4.0 / 3.0))
        #-------------------------------------------------#
        # Test that the symmetry conditions are executed. #
        #-------------------------------------------------#
        obj = GlorotBilinearForm(initializer=lasagne.init.Uniform,
                                 gain=1.0,
                                 symmetry=GlorotBilinearForm.ANTISYMMETRIC)
        self.assertRaises(ValueError, (lambda: obj.sample((4, 3, 2))))
        sample = GlorotBilinearForm(
            initializer=lasagne.init.Uniform,
            gain=1.0,
            symmetry=GlorotBilinearForm.ANTISYMMETRIC).sample((3, 3, 4))
        self.assertEqual(np.sum(sample + np.swapaxes(sample, 0, 1)), 0.0)
        sample = GlorotBilinearForm(
            initializer=lasagne.init.Uniform,
            gain=1.0,
            symmetry=GlorotBilinearForm.ANTISYMMETRIC).sample((3, 3, 4))
        self.assertEqual(np.sum(sample - np.swapaxes(sample, 0, 1)), 0.0)


if __name__ == '__main__':
    unittest.main()
