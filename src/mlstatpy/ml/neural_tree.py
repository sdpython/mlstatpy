# -*- coding: utf-8 -*-
"""
@file
@brief Conversion from tree to neural network.
"""
import numpy
import numpy.random as rnd
from scipy.special import expit  # pylint: disable=E0611


class NeuralTreeNode:
    """
    One node in a neural network.
    """

    @staticmethod
    def _relu(x):
        return x if x > 0 else 0

    @staticmethod
    def get_activation_function(activation):
        """
        Returns the activation function.
        """
        if activation in {'logistic', 'expit'}:
            return expit
        if activation == 'relu':
            return NeuralTreeNode._relu
        if activation == 'identity':
            return lambda x: x
        raise ValueError(
            "Unknown activation function '{}'.".format(activation))

    def __init__(self, weights, bias=None, activation='sigmoid', nodeid=-1):
        """
        @param      weights     weights
        @param      bias        bias, if None, draws a random number
        @param      activation  activation function
        @param      nodeid      node id
        """
        if bias is None:
            bias = rnd.randn()
        if isinstance(weights, int):
            weights = rnd.randn(weights)
        self.coef = numpy.empty(len(weights) + 1)
        self.coef[1:] = weights
        self.coef[0] = bias
        self.activation = activation
        self.activation_ = NeuralTreeNode.get_activation_function(activation)
        self.nodeid = nodeid

    @property
    def weights(self):
        "Returns the weights."
        return self.coef[1:]

    @property
    def bias(self):
        "Returns the weights."
        return self.coef[0]

    def __getstate__(self):
        "usual"
        return {
            'coef': self.coef, 'activation': self.activation,
            'nodeid': self.nodeid}

    def __setstate__(self, state):
        "usual"
        self.coef = state['coef']
        self.activation = state['activation']
        self.nodeid = state['nodeid']
        self.activation_ = NeuralTreeNode.get_activation_function(
            self.activation)

    def __eq__(self, obj):
        if self.coef.shape != obj.coef.shape:
            return False
        if any(map(lambda xy: xy[0] != xy[1], zip(self.coef, obj.coef))):
            return False
        if self.activation != obj.activation:
            return False
        return True

    def __repr__(self):
        "usual"
        return "%s(weights=%r, bias=%r, activation=%r)" % (
            self.__class__.__name__, self.coef[1:],
            self.coef[0], self.activation)

    def predict(self, X):
        "Computes neuron outputs."
        return self.activation_(X @ self.coef[1:] + self.coef[0])

    @property
    def ndim(self):
        "Returns the input dimension."
        return self.coef.shape[0] - 1


class NeuralTreeNet:
    """
    Node ensemble.
    """

    def __init__(self, dim):
        """
        @param      dim     space dimension
        """
        self.dim = dim
        self.nodes = [
            NeuralTreeNode(
                numpy.ones((dim,), dtype=numpy.float64),
                bias=numpy.float64(0.),
                activation='identity', nodeid=0)]
        self.nodes_attr = [dict(inputs=numpy.arange(0, dim), output=dim)]
        self._update_members()

    def _update_members(self):
        self.size_ = max(d['output'] for d in self.nodes_attr) + 1

    def __repr__(self):
        "usual"
        return "%s(%d)" % (self.__class__.__name__, self.dim)

    def append(self, node, inputs):
        """
        Appends a node into the graph.

        @param      node        node to add
        @param      inputs      index of input nodes
        """
        if len(inputs) != node.ndim:
            raise ValueError('Node and inputs must have the same dimension.')
        node.nodeid = len(self.nodes)
        self.nodes.append(node)
        attr = dict(inputs=numpy.array(inputs), output=self.size_)
        self.nodes_attr.append(attr)
        self.size_ += 1

    def _predict_one(self, X):
        res = numpy.zeros((self.size_,), dtype=numpy.float64)
        res[:self.dim] = X
        for node, attr in zip(self.nodes, self.nodes_attr):
            res[attr['output']] = node.predict(res[attr['inputs']])
        return res

    def predict(self, X):
        if len(X.shape) == 2:
            res = numpy.zeros((X.shape[0], self.size_))
            for i, x in enumerate(X):
                res[i, :] = self._predict_one(x)
            return res
        return self._predict_one(X)
