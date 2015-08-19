'''
| Filename    : lib_score_builder.py
| Description : Library of Score Builder Objects.
| Author      : Pushpendre Rastogi
| Created     : Sun Aug 16 17:28:50 2015 (-0400)
| Last-Updated: Wed Aug 19 03:08:46 2015 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 30
The guiding principle for this library is that classes should be closed
for modification but open for extension.
'''
import abc, re
import pylearn2
import lasagne.init
import theano.tensor
import lasagne.nonlinearities
from lasagne_extension import NeuralTensorNetworkLayer, GlorotBilinearForm

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
        if symmetry_flag == '':
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
            pass
        else:
            activation = self.nn_activation.get_output_for(
                theano.tensor.concatenate([left_input, right_input], axis=1))
        if self.add_tensor_activation:
            activation += self.ntn_activation.get_output_for(
                theano.tensor.concatenate([left_input, right_input], axis=1))
        return activation

    def __call__(self):
        raise NotImplementedError

class ScoreBuilder(pylearn2.models.model.Model):

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
    __metaclass__ = abc.ABCMeta
    SYMMETRIC = 'Symmetric'
    ANTISYMMETRIC = 'AntiSymmetric'
    SYMMETRY_FLAGS = [SYMMETRIC, ANTISYMMETRIC, '']

    @abc.abstractmethod
    def __init__(self,
                 input_dim=25,
                 activation_units=80,
                 dropout_p=0, # 0.2,
                 nonlinearity=lasagne.nonlinearities.leaky_rectify,
                 optional_layer_num_units=80,
                 optional_layer_nonlinearity=lasagne.nonlinearities.tanh,
                 final_chromaticity=1,
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
        #---------------------------------------------------------------#
        # Parse the derived class name to convert it actionable options #
        #---------------------------------------------------------------#
        (symmetry_flag, optional_layer_flag, score_function_flag) = (
            self.parse_class_name_for_flags())
        assert optional_layer_flag in ['OptionalLayer', '']
        assert score_function_flag in ['NN', 'NTN']
        self._params = []
        add_tensor_activation = (True
                                 if score_function_flag == 'NTN'
                                 else False)
        #-----------------------------------#
        # Control addition of OptionalLayer #
        #-----------------------------------#
        if optional_layer_flag == 'OptionalLayer':
            self.optional_layer = OptionalLayer(
                input_dim,
                optional_layer_nonlinearity,
                optional_layer_num_units)
            self._params.extend(self.optional_layer.get_params())
            dim_after_optional_layer = optional_layer_num_units
        else:
            self.optional_layer = PassThroughLayer()
            dim_after_optional_layer = input_dim
        #-------------------------------------------------------#
        # Build comparator based on score_function and symmetry #
        #-------------------------------------------------------#
        self.comparator_activation_layer = ComparatorActivation(
            dim_after_optional_layer, dim_after_optional_layer,
            symmetry_flag, add_tensor_activation, activation_units)
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

    def parse_class_name_for_flags(self):
        derived_class_name = self.__class__.__name__
        (symmetry_flag, optional_layer_flag, score_function_flag) = re.findall(
            '(Symmetric|AntiSymmetric)?(OptionalLayer)?(NN|NTN)ScoreBuilder',
            derived_class_name)[0]
        return (symmetry_flag, optional_layer_flag, score_function_flag)

    def get_output_for(self, node1_batch, node2_batch):
        (node1_batch, node2_batch) = self.optional_layer.get_output_for(
            node1_batch, node2_batch)
        activation = self.comparator_activation_layer.get_output_for(
            node1_batch, node2_batch)
        activation = self.dropout_layer.get_output_for(activation)
        activation = self.nonlinearity(activation)
        activation = self.final_score_layer.get_output_for(activation)
        return activation

class NNScoreBuilder(ScoreBuilder):
    def __init__(self, *args, **kwargs):
        super(NNScoreBuilder, self).__init__(*args, **kwargs)


class OptionalLayerNNScoreBuilder(ScoreBuilder):
    def __init__(self, *args, **kwargs):
        super(OptionalLayerNNScoreBuilder, self).__init__(*args, **kwargs)


class NTNScoreBuilder(ScoreBuilder):
    def __init__(self, *args, **kwargs):
        super(NTNScoreBuilder, self).__init__(*args, **kwargs)


class OptionalLayerNTNScoreBuilder(ScoreBuilder):
    def __init__(self, *args, **kwargs):
        super(OptionalLayerNTNScoreBuilder, self).__init__(*args, **kwargs)


class SymmetricNNScoreBuilder(ScoreBuilder):
    def __init__(self, *args, **kwargs):
        super(SymmetricNNScoreBuilder, self).__init__(*args, **kwargs)


class AntiSymmetricNNScoreBuilder(ScoreBuilder):
    def __init__(self, *args, **kwargs):
        super(AntiSymmetricNNScoreBuilder, self).__init__(*args, **kwargs)


class SymmetricNTNScoreBuilder(ScoreBuilder):
    def __init__(self, *args, **kwargs):
        super(SymmetricNTNScoreBuilder, self).__init__(*args, **kwargs)


class AntiSymmetricNTNScoreBuilder(ScoreBuilder):
    def __init__(self, *args, **kwargs):
        super(AntiSymmetricNTNScoreBuilder, self).__init__(*args, **kwargs)

import unittest
class TestNNScoreBuilder(unittest.TestCase):
    def test_circuit(self):

        pass

class TestOptionalLayerNNScoreBuilder(unittest.TestCase):
    def test_circuit(self):
        pass

class TestSymmetricNNScoreBuilder(unittest.TestCase):
    def test_circuit(self):
        pass

class TestNTNScoreBuilder(unittest.TestCase):
    def test_circuit(self):
        pass

class TestSymmetricNTNScoreBuilder(unittest.TestCase):
    def test_circuit(self):
        pass

class TestAntiSymmetricNTNScoreBuilder(unittest.TestCase):
    def test_circuit(self):
        pass

if __name__ == '__main__':
    unittest.main()
