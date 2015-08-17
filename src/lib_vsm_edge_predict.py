# -*- coding: utf-8 -*-
'''
| Filename    : lib_vsm_edge_predict.py
| Description : A library of vector space models for predicting edges in a graph.
| Author      : Pushpendre Rastogi
| Created     : Sat Aug 15 21:25:12 2015 (-0400)
| Last-Updated:
|           By:
|     Update #: 5
'''
# your model maps an input to an output, the output is compared with
# some ground truth using some measure of dissimilarity, and the
# parameters of the model are changed to reduce this measure using
# gradient information.
import abc
from pylearn2.models.model import Model
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX
import lasagne


class MyCostSubclass(DefaultDataSpecsMixin, Cost):
    # Here it is assumed that we are doing supervised learning
    supervised = True

    def expr(self, model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)

        inputs, targets = data
        outputs = model.some_method_for_outputs(inputs)
        loss =  # some loss measure involving outputs and targets
        return loss


class SingleNodeEmbeddingEdgePredictor(Model):

    def __init__(self, score_builder_list, node_count=10, chromaticity=7, node_embedding_dim=80):
        '''
        Params
        ------
        score_builder_list : A list of functions. Each function returns a circuit
          that can map 2 input vectors and an Edge Category to a real number;
          the score of the edge.
        node_count : The number of nodes in the graph.
        chromaticity : The number of edge types or categories in the graph.
          It can also be interpreted as the colors of edges in a graph.
        node_embedding_dim : The dimensionality of the nodes.
        '''
        super(NNEdgePredictor, self).__init__()
        # Take the
        self._params = [
            # List of all the model parameters
        ]

        self.input_space =  # Some `pylearn2.space.Space` subclass
        self.output_space =  # Some `pylearn2.space.Space` subclass
        return

    def some_method_for_outputs(self, inputs):
        # Some computation involving the inputs
        pass

import unittest


class TestSingleNodeEmbeddingEdgePredictor(unittest.TestCase):

    def test_single_node_embedding_edge_predictor(self):
        score_builder = ScoreBuilder()
        obj = SingleNodeEmbeddingEdgePredictor(score_builder,
                                               node_count=10,
                                               chromaticity=3,
                                               node_embedding_dim=10)
        obj.override_parameters =
        self.assertEqual(len(obj._params), None)


if __name__ == '__main__':
    unittest.main()
