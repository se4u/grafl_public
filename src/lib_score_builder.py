'''
| Filename    : lib_score_builder.py
| Description : Library of Score Builder Objects.
| Author      : Pushpendre Rastogi
| Created     : Sun Aug 16 17:28:50 2015 (-0400)
| Last-Updated: Wed Aug 19 21:39:43 2015 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 93
The guiding principle for this library is that classes should be closed
for modification but open for extension.
'''
from pylearn2.models import Model
import pylearn2.space
import lasagne.init
import theano.tensor
import lasagne.nonlinearities
from lasagne.layers import EmbeddingLayer
from lasagne_extension import NeuralTensorNetworkLayer, GlorotBilinearForm
import numpy
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin


def score_builder_to_glorot_convert(symmetry):
    assert symmetry in ScoreBuilder.SYMMETRY_FLAGS
    return (GlorotBilinearForm.SYMMETRIC
            if symmetry == ScoreBuilder.SYMMETRIC
            else (GlorotBilinearForm.ANTISYMMETRIC
                  if symmetry == ScoreBuilder.ANTISYMMETRIC
                  else GlorotBilinearForm.NEITHERSYMMETRIC))


def get_ntn_activation(input_shape, bifurcation_point, num_output_units, symmetry):
    """
    Params
    ------
    input_shape : tuple of ints
        The first element of input_shape may be `None`. It would refer to the batch size.

    num_output_units : int

    nonlinearity : elemwise real function
        A member of the lasagne.nonlinearities module.
    """
    assert isinstance(input_shape, tuple)
    assert symmetry in ScoreBuilder.SYMMETRY_FLAGS
    symmetry = score_builder_to_glorot_convert(symmetry)
    return NeuralTensorNetworkLayer(
        incoming=input_shape,
        num_units=num_output_units,
        bifurcation_point=bifurcation_point,
        nonlinearity=lasagne.nonlinearities.identity,
        W=None,
        b=None,
        T=GlorotBilinearForm(symmetry=symmetry))


def get_nn_layer(input_shape, num_output_units, nonlinearity):
    """
    Params
    ------
    input_shape : tuple of ints
        The first element of input_shape may be `None`. It would refer to the batch size.

    num_output_units : int

    nonlinearity : elemwise real function
        A member of the lasagne.nonlinearities module.
    """
    assert isinstance(input_shape, tuple)
    return NeuralTensorNetworkLayer(
        input_shape,
        num_output_units,
        bifurcation_point=None,
        nonlinearity=nonlinearity,
        W=lasagne.init.GlorotUniform(),
        b=lasagne.init.Constant(),
        T=None)


class PassThroughLayer(object):

    @staticmethod
    def get_output_for(*args):
        return (args[0] if len(args) == 1 else args)


class OptionalLayer(object):

    def __init__(self, input_dim, nonlinearity, num_output_units):
        self.layer_left_vertex = get_nn_layer(
            (None, input_dim),
            num_output_units,
            nonlinearity)
        self.layer_right_vertex = get_nn_layer(
            (None, input_dim),
            num_output_units,
            nonlinearity)

    def get_params(self):
        return (self.layer_left_vertex.get_params()
                + self.layer_right_vertex.get_params())

    def get_output_for(self, left_input, right_input):
        return (self.layer_left_vertex.get_output_for(left_input),
                self.layer_right_vertex.get_output_for(right_input))

    def __call__(self):
        raise NotImplementedError


class ComparatorActivation(object):

    def __init__(self, left_input_dim, right_input_dim,
                 symmetry_flag, add_tensor_activation, num_output_units):
        """
        Params
        ------
        symmetry_flag         --
        add_tensor_activation --
        num_units             --
        """
        assert symmetry_flag in ScoreBuilder.SYMMETRY_FLAGS
        assert left_input_dim == right_input_dim
        if symmetry_flag == ScoreBuilder.NEITHERSYMMETRIC:
            nn_input_dim = left_input_dim + right_input_dim
        else:
            nn_input_dim = left_input_dim
        self.nn_activation = get_nn_layer(
            (None, nn_input_dim),
            num_output_units,
            lasagne.nonlinearities.identity)
        self._params = self.nn_activation.get_params()
        if add_tensor_activation:
            self.ntn_activation = get_ntn_activation(
                (None, left_input_dim + right_input_dim),
                left_input_dim,
                num_output_units,
                symmetry_flag)
            self._params.extend(self.ntn_activation.get_params())
        self.symmetry_flag = symmetry_flag
        self.add_tensor_activation = add_tensor_activation

    def get_params(self):
        return self._params

    def get_output_for(self, left_input, right_input):
        if self.symmetry_flag == ScoreBuilder.SYMMETRIC:
            activation = self.nn_activation.get_output_for(
                left_input + right_input)
        elif self.symmetry_flag == ScoreBuilder.ANTISYMMETRIC:
            activation = self.nn_activation.get_output_for(
                left_input - right_input)
        else:
            activation = self.nn_activation.get_output_for(
                theano.tensor.concatenate([left_input, right_input], axis=1))
        if self.add_tensor_activation:
            activation += self.ntn_activation.get_output_for(
                theano.tensor.concatenate([left_input, right_input], axis=1))
        return activation

    def __call__(self):
        raise NotImplementedError


class ScoreBuilder(Model):

    ''' Create computational graphs representing the following sequential
    operations.

    Use a Softmax Classifier
    ------------------------
    Apply tanh / sigmoid / leaky relu / relu
    ----------------------------------------
    Multiply with a Matrix (output dim=100)
    + Add Bias
    + Add a bilinear tensor form
    --------------------------------------------------------------------
    Transform the resultant vector through tanh / relu layer (OPTIONAL)
    --------------------------------------------------------------------
    Dropout
    ----------------------------
    Concatenate/ Add / Substract

    Provide functions to
    - Serialize
    - Initialize
    - Load
    the parameters necessary for the models.
    '''
    SYMMETRIC = 'Symmetric'
    ANTISYMMETRIC = 'AntiSymmetric'
    NEITHERSYMMETRIC = None
    SYMMETRY_FLAGS = [SYMMETRIC, ANTISYMMETRIC, NEITHERSYMMETRIC]

    def __init__(self,
                 input_categories=10,
                 input_dim=25,
                 activation_units=80,
                 dropout_p=0,  # 0.2,
                 nonlinearity=lasagne.nonlinearities.leaky_rectify,
                 optional_layer_num_units=80,
                 optional_layer_nonlinearity=lasagne.nonlinearities.tanh,
                 final_chromaticity=1,
                 symmetry_flag=None,
                 add_optional_layer=False,
                 add_tensor_activation=False,
                 **kwargs):
        """
        Params
        ------

        dropout_p : float
            The dropout probability to use in the dropout layer. If dropout_p < 0
            then dropout is not done.

        nonlinearity : :subclass:`lasagne.nonlinearities`
            The nonlinearity used in the comparator function before producing the final
            score.

        optional_layer_num_units : int
            The dimensionality of the optional nonlinear tranformation of the vertex
            embeddings.

        optional_layer_nonlinearity : :subclass:`lasagne.nonlinearities`
            The nonlinearity employes in the optional nonlinear transformations of the
            vertex embeddings
        """
        super(ScoreBuilder, self).__init__(**kwargs)
        self.activation_units = activation_units
        self.dropout_p = dropout_p
        self.nonlinearity = nonlinearity
        self.optional_layer_num_units = optional_layer_num_units
        self.optional_layer_nonlinearity = optional_layer_nonlinearity
        assert symmetry_flag in self.SYMMETRY_FLAGS
        self.symmetry_flag = symmetry_flag
        self.add_optional_layer = add_optional_layer
        self._params = []
        self.add_tensor_activation = add_tensor_activation
        #---------------------------------------------------#
        # Declare the input and output spaces of this model #
        #---------------------------------------------------#
        import pdb
        # pdb.set_trace()
        self.input_space = pylearn2.space.IndexSpace(
            max_labels=input_categories, dim=2, dtype='int32')
        self.output_space = pylearn2.space.IndexSpace(
            max_labels=final_chromaticity, dim=2, dtype='int32')
        #-----------------------------#
        # Instantiate Embedding Layer #
        #-----------------------------#
        self.embedding_layer = EmbeddingLayer(
            (None, None), input_categories, input_dim)
        self._params.extend(self.embedding_layer.get_params())
        #-----------------------------------#
        # Control addition of OptionalLayer #
        #-----------------------------------#
        if add_optional_layer:
            self.optional_layer = OptionalLayer(
                input_dim=input_dim,
                nonlinearity=optional_layer_nonlinearity,
                num_output_units=optional_layer_num_units)
            self._params.extend(self.optional_layer.get_params())
            dim_after_optional_layer = optional_layer_num_units
        else:
            self.optional_layer = PassThroughLayer()
            dim_after_optional_layer = input_dim
        #-------------------------------------------------------#
        # Build comparator based on score_function and symmetry #
        #-------------------------------------------------------#
        self.comparator_activation_layer = ComparatorActivation(
            left_input_dim=dim_after_optional_layer,
            right_input_dim=dim_after_optional_layer,
            symmetry_flag=symmetry_flag,
            add_tensor_activation=add_tensor_activation,
            num_output_units=activation_units)
        self._params.extend(self.comparator_activation_layer.get_params())
        #-----------------------------------------------------#
        # Dropout -> Nonlinearity -> Score through Projection #
        #-----------------------------------------------------#
        self.dropout_layer = (
            lasagne.layers.DropoutLayer((None, ), p=self.dropout_p)
            if self.dropout_p > 0
            else PassThroughLayer())
        self.nonlinearity = nonlinearity
        self.final_score_layer = NeuralTensorNetworkLayer(
            (None, activation_units),
            final_chromaticity,
            bifurcation_point=None,
            nonlinearity=lasagne.nonlinearities.identity,
            W=lasagne.init.GlorotUniform(),
            b=None,
            T=None)
        self._params.extend(self.final_score_layer.get_params())

    def get_params(self):
        return self._params

    def get_output_for(self, input_batch):
        node1_batch = input_batch[:, 0]
        node2_batch = input_batch[:, 1]
        node1_batch = self.embedding_layer.get_output_for(node1_batch)
        node2_batch = self.embedding_layer.get_output_for(node2_batch)
        (node1_batch, node2_batch) = self.optional_layer.get_output_for(
            node1_batch, node2_batch)
        activation = self.comparator_activation_layer.get_output_for(
            node1_batch, node2_batch)
        activation = self.dropout_layer.get_output_for(activation)
        activation = self.nonlinearity(activation)
        activation = self.final_score_layer.get_output_for(activation)
        return activation

    def __setattr__(self, name, value):
        if name == 'monitor':
            import pdb
            # pdb.set_trace()
        super(ScoreBuilder, self).__setattr__(name, value)


class MultiClassSoftmaxCrossEntropyCost(DefaultDataSpecsMixin, Cost):
    supervised = True

    def expr(self, model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)
        # Compute the Cross-Entropy Cost
        inputs, targets = data
        output_predicted_prob = theano.tensor.nnet.softmax(
            model.get_output_for(inputs))
        loss = - ((targets * theano.tensor.log(output_predicted_prob)
                   ).astype('float32')).sum(axis=1, acc_dtype='float32')
        return loss.mean()

import unittest


class TestModuleFunctions(unittest.TestCase):

    def test_score_builder_to_glorot_convert(self):
        self.assertEqual(
            score_builder_to_glorot_convert(ScoreBuilder.ANTISYMMETRIC),
            GlorotBilinearForm.ANTISYMMETRIC)
        self.assertEqual(
            score_builder_to_glorot_convert(ScoreBuilder.SYMMETRIC),
            GlorotBilinearForm.SYMMETRIC)
        self.assertEqual(
            score_builder_to_glorot_convert(ScoreBuilder.NEITHERSYMMETRIC),
            GlorotBilinearForm.NEITHERSYMMETRIC)

    def test_get_ntn_activation(self):
        obj = get_ntn_activation(
            (None, 10), 6, 3, ScoreBuilder.NEITHERSYMMETRIC)
        obj_params = obj.get_params()
        self.assertEqual(len(obj_params), 1)
        T = obj_params[0].eval()
        self.assertEqual(T.shape, (6, 4, 3))
        test_input = numpy.array([range(10)])
        output = obj.get_output_for(test_input).eval()
        expected_activation = NeuralTensorNetworkLayer.tensor_matrix_activation(
            T, test_input, 6, numpy.tensordot)
        numpy.testing.assert_array_almost_equal(output, expected_activation)

    def test_get_nn_layer(self):
        obj = get_nn_layer((None, 10), 3, lasagne.nonlinearities.identity)
        obj_params = obj.get_params()
        self.assertEqual(len(obj_params), 2)
        W = obj_params[0].eval()
        b = obj_params[1].eval()
        self.assertEqual(W.shape, (10, 3))
        test_input = numpy.array([range(10)])
        output = obj.get_output_for(test_input).eval()
        expected_activation = numpy.dot(test_input, W) + b
        numpy.testing.assert_array_almost_equal(output, expected_activation)


class TestPassThroughLayer(unittest.TestCase):

    def test_get_output_for(self):
        obj = PassThroughLayer()
        self.assertEqual(1, obj.get_output_for(1))
        self.assertEqual((1, 2), obj.get_output_for(1, 2))


class TestComparatorActivation(unittest.TestCase):

    def obj_param_helper(self, symmetry, add_tensor):
        obj = ComparatorActivation(3, 3, symmetry, add_tensor, 2)
        params = obj.get_params()
        return (obj, params)

    def test_get_output_for(self):
        # Tensor, Symmetry
        (obj, params) = self.obj_param_helper(ScoreBuilder.SYMMETRIC, True)
        self.assertEqual(len(params), 3)
        self.assertEqual(params[-1].eval().shape, (3, 3, 2))
        self.assertEqual(params[0].eval().shape, (3, 2))
        # Tensor, No Symmetry
        (obj, params) = self.obj_param_helper(
            ScoreBuilder.NEITHERSYMMETRIC, True)
        self.assertEqual(len(params), 3)
        self.assertEqual(params[-1].eval().shape, (3, 3, 2))
        self.assertEqual(params[0].eval().shape, (6, 2))
        self.assertEqual(params[1].eval().shape, (2,))
        # No Tensor, Symmetry
        (obj, params) = self.obj_param_helper(ScoreBuilder.SYMMETRIC, False)
        self.assertEqual(len(params), 2)
        self.assertEqual(params[0].eval().shape, (3, 2))
        self.assertEqual(params[1].eval().shape, (2,))
        random_input = numpy.array([[1, 2, 3]], dtype='float32')
        numpy.testing.assert_array_equal(
            obj.get_output_for(random_input, -random_input).eval(),
            numpy.zeros((1, 2), dtype='float32'))

        # No Tensor, No Symmetry
        (obj, params) = self.obj_param_helper(
            ScoreBuilder.NEITHERSYMMETRIC, False)
        self.assertEqual(len(params), 2)
        self.assertEqual(params[0].eval().shape, (6, 2))
        self.assertEqual(params[1].eval().shape, (2,))

if __name__ == '__main__':
    unittest.main()
