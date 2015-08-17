'''
| Filename    : lib_score_builder.py
| Description : Library of Score Builder Objects.
| Author      : Pushpendre Rastogi
| Created     : Sun Aug 16 17:28:50 2015 (-0400)
| Last-Updated: Mon Aug 17 02:04:23 2015 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 4
The guiding principle for this library is that classes should be closed
for modification but open for extension.
'''
import lasagne
import numpy


class ScoreBuilder(object):

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

    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def circuit(self, node1_batch, node2_batch):
        ''' This method uses the name of the subclass to figure out the
        implementation used. The format is as follows
        [Symmetric|AntiSymmetric|][OptionalLayer|][NN|NTN]ScoreBuilder
        3 x 2 x 2 = 12 classes.

        NOTE: We do this instead of passing these as parameters because
        Explicit names need to be maintained anyway, as they are clearer
        and using parameters then just duplicates code. This way is DRY.
        '''
        #---------------------------------------------------------------#
        # Parse the derived class name to convert it actionable options #
        #---------------------------------------------------------------#
        derived_class_name = self.__class__.__name__
        (symmetry_flag, optional_layer_flag, score_function_flag) = re.findall(
            '(Symmetric|AntiSymmetric)?(OptionalLayer)?(NN|NTN)ScoreBuilder',
            derived_class_name)[0]

        if optional_layer_flag == 'OptionalLayer':
            (node1_batch, node2_batch) = self.pass_through_nonlinearity(
                node1_batch, node2_batch)
        elif optional_layer_flag == '':
            pass
        else:
            raise RuntimeError

        if score_function_flag == 'NN':
            circuit = self.nn_helper_circuit(
                node1_batch, node2_batch, symmetry_flag)
        elif score_function_flag == 'NTN':
            circuit = self.ntn_helper_circuit(
                node1_batch, node2_batch, symmetry_flag)
        else:
            raise RuntimeError

    def ntn_helper_circuit(self, node1_batch, node2_batch, symmetry_flag):
        assert symmetry_flag in ['Symmetric', 'AntiSymmetric', '']
        # Neural Part
        neural_part = self.process_nn_circuit_input(
            node1_batch, node2_batch, symmetry_flag)
        neural_part = lasagne.layers.DenseLayer(
            neural_part,
            num_units=80,
            nonlinearity=lasagne.nonlinearities.identity,
            W=lasagne.init.GlorotUniform(),
            b=lasagne.init.Constant())
        # Tensor Part
        tensor_part = lasagne.layers.
        # Add Neural and Tensor Part

        # Do Dropout

        pass

    def process_nn_circuit_input(self, node1_batch, node2_batch, symmetry_flag):
        return (lasagne.layers.ConcatLayer([node1_batch, node2_batch], axis=1)
                if symmetry_flag == ''
                else ((node1_batch + node2_batch)
                      if symmetry_flag == 'Symmetric'
                      else (node1_batch - node2_batch)))

    def nn_helper_circuit(self, node1_batch, node2_batch, symmetry_flag):
        assert symmetry_flag in ['Symmetric', 'AntiSymmetric', '']
        l = self.process_nn_circuit_input(
            node1_batch, node2_batch, symmetry_flag)
        l = lasagne.layers.DropoutLayer(l, p=0.2)
        l = lasagne.layers.DenseLayer(l,
                                      num_units=80,
                                      nonlinearity=lasagne.nonlinearities.rectify,
                                      W=lasagne.init.GlorotUniform(),
                                      b=lasagne.init.Constant())
        l = lasagne.layers.DenseLayer(l,
                                      num_units=self.chromaticity,
                                      nonlinearity=lasagne.nonlinearities.softmax,
                                      W=lasagne.init.GlorotUniform())
        return l

    def pass_through_nonlinearity(self, node1_batch, node2_batch,
                                  nonlinearity=lasagne.nonlinearities.tanh):
        optional_layer = (lambda layer:
                          lasagne.layers.DenseLayer(
                              layer,
                              num_units=80,
                              nonlinearity=nonlinearity,
                              W=lasagne.init.GlorotUniform(),
                              b=lasagne.init.Constant()))

        node1_batch = optional_layer(node1_batch)
        node2_batch = optional_layer(node2_batch)


class NNScoreBuilder(ScoreBuilder):

    def circuit(self, node1_batch, node2_batch):

        return self.helper_circuit_1(node1_batch, node2_batch)


class OptionalLayerNNScoreBuilder(ScoreBuilder):

    def circuit(self, node1_batch, node2_batch):
        (node1_batch, node2_batch) = self.pass_through_nonlinearity(
            node1_batch, node2_batch)
        return self.helper_circuit_1(node1_batch, node2_batch)


class NTNScoreBuilder(ScoreBuilder):

    def circuit(self):
        pass


class OptionalLayerNTNScoreBuilder(ScoreBuilder):

    def __init__(self):
        pass

    def circuit(self):
        pass


class SymmetricNNScoreBuilder(ScoreBuilder):

    def __init__(self):
        pass


class AntiSymmetricNNScoreBuilder(ScoreBuilder):

    def __init__(self):
        pass


class SymmetricNTNScoreBuilder(ScoreBuilder):

    def __init__(self):
        pass


class AntiSymmetricNTNScoreBuilder(ScoreBuilder):

    def __init__(self):
        pass


import unittest


class TestNNScoreBuilder(unittest.TestCase):

    def test_circuit(self):
        pass


class TestOptionalLayerNNScoreBuilder(unittest.TestCase):

    def test_circuit(self):
        pass


class TestNTNScoreBuilder(unittest.TestCase):

    def test_circuit(self):
        pass


class TestOptionalLayerNTNScoreBuilder(unittest.TestCase):

    def test_circuit(self):
        pass

if __name__ == '__main__':
    unittest.main()
