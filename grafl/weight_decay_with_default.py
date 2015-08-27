'''
| Filename    : weight_decay_with_default.py
| Description : WeightDecayWithDefault class allows specifying weight decay for all layers that support it.
| Author      : Pushpendre Rastogi
| Created     : Mon Aug 24 17:45:46 2015 (-0400)
| Last-Updated: Mon Aug 24 19:23:09 2015 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 10
'''
from functools import wraps
import operator
import warnings
import theano.tensor
from pylearn2.costs.cost import Cost, NullDataSpecsMixin
from pylearn2.utils.exc import reraise_as


class WeightDecayWithDefault(NullDataSpecsMixin, Cost):

    """L2 regularization cost for MLP.

    coeff * sum(sqr(weights)) for each set of weights.

    Parameters
    ----------
    coeffs : dict
        Dictionary with layer names as its keys,
        specifying the coefficient to multiply
        with the cost defined by the squared L2 norm of the weights for
        each layer.

        Each element may in turn be a list, e.g., for CompositeLayers.
    """

    def __init__(self, coeffs=None, default=5e-5):
        self.coeffs = ({}
                       if coeffs is None
                       else coeffs)
        self.default = default

    def expr(self, model, data, ** kwargs):
        """Returns a theano expression for the cost function.

        Parameters
        ----------
        model : MLP
        data : tuple
            Should be a valid occupant of
            CompositeSpace(model.get_input_space(),
            model.get_output_space())

        Returns
        -------
        total_cost : theano.gof.Variable
            coeff * sum(sqr(weights))
            added up for each set of weights.
        """
        self.get_data_specs(model)[0].validate(data)
        assert theano.tensor.scalar(
        ) != 0.  # make sure theano semantics do what I want

        def wrapped_layer_cost(layer, coeff):
            try:
                return layer.get_weight_decay(coeff)
            except NotImplementedError:
                if coeff == 0.:
                    return 0.
                else:
                    reraise_as(NotImplementedError(str(type(layer)) +
                                                   " does not implement "
                                                   "get_weight_decay."))

        assert not isinstance(self.coeffs, list)
        layer_costs = []
        for layer in model.layers:
            layer_name = layer.layer_name
            try:
                coeff = self.coeffs[layer_name]
            except KeyError:
                warnings.warn("Adding weight decay to %s layer" % layer_name)
                coeff = self.default
            cost = wrapped_layer_cost(layer, coeff)
            if cost != 0.:
                layer_costs.append(cost)

        if len(layer_costs) == 0:
            rval = theano.tensor.as_tensor_variable(0.)
            rval.name = '0_weight_decay'
            return rval
        else:
            total_cost = reduce(operator.add, layer_costs)
        total_cost.name = 'MLP_WeightDecay'

        assert total_cost.ndim == 0

        total_cost.name = 'weight_decay'

        return total_cost

    @wraps(Cost.is_stochastic)
    def is_stochastic(self):
        return False
