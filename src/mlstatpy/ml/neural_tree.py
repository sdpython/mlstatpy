# -*- coding: utf-8 -*-
"""
@file
@brief Conversion from tree to neural network.
"""
import numpy
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

    def __init__(self, weights, bias, activation='sigmoid'):
        """
        @param      weights     weights
        @param      bias        bias
        @param      activation  activation function
        """
        self.coef = numpy.empty(len(weights) + 1)
        self.coef[1:] = weights
        self.coef[0] = bias
        self.activation = activation
        self.activation_ = NeuralTreeNode.get_activation_function(activation)

    def __getstate__(self):
        "usual"
        return {'coef': self.coef, 'activation': self.activation}

    def __setstate__(self, state):
        "usual"
        self.coef = state['coef']
        self.activation = state['activation']
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
