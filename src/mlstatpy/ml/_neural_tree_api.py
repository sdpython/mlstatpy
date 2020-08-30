# -*- coding: utf-8 -*-
"""
@file
@brief Conversion from tree to neural network.
"""
import numpy
from ..optim import SGDOptimizer


class _TrainingAPI:
    """
    Declaration of function needed to train a model.
    """

    @property
    def training_weights(self):
        "Returns the weights."
        raise NotImplementedError(  # pragma: no cover
            "This should be overwritten.")

    def update_training_weights(self, grad, add=True):
        """
        Updates weights.

        :param grad: vector to add to the weights such as gradient
        :param add: addition or replace
        """
        raise NotImplementedError(  # pragma: no cover
            "This should be overwritten.")

    def fill_cache(self, X):
        """
        Creates a cache with intermediate results.
        """
        return None  # pragma: no cover

    def loss(self, X, y, cache=None):
        """
        Computes the loss. Returns a float.
        """
        raise NotImplementedError(  # pragma: no cover
            "This should be overwritten.")

    def losss(self, X, y, cache=None):
        """
        Computes the loss due to prediction error. Returns a float.
        """
        raise NotImplementedError(  # pragma: no cover
            "This should be overwritten.")

    def lossw(self, X, y, cache=None):
        """
        Computes the loss due to regularization. Returns a float.
        """
        raise NotImplementedError(  # pragma: no cover
            "This should be overwritten.")

    def dlossds(self, X, y, cache=None):
        """
        Computes the loss derivative due to prediction error.
        """
        raise NotImplementedError(  # pragma: no cover
            "This should be overwritten.")

    def dlossdw(self):
        """
        Computes the loss derivative due to regularization.
        """
        raise NotImplementedError(  # pragma: no cover
            "This should be overwritten.")

    def gradient_backward(self, graddx, graddw, X, inputs=False, cache=None):
        """
        Computes the gradient in X.

        :param graddx: existing gradient against the inputs
        :param graddw: existing gradient against the weights
        :param X: computes the gradient in X
        :param inputs: if False, derivative against the coefficients,
            otherwise against the inputs.
        :param cache: cache intermediate results to avoid more computation
        :return: gradient
        """
        raise NotImplementedError(  # pragma: no cover
            "This should be overwritten.")

    def gradient(self, X, y, inputs=False):
        """
        Computes the gradient in *X* knowing the expected value *y*.

        :param X: computes the gradient in X
        :param y: expected values
        :param inputs: if False, derivative against the coefficients,
            otherwise against the inputs.
        :return: gradient
        """
        if len(X.shape) != 1:
            raise ValueError(  # pragma: no cover
                "X must a vector of one dimension but has shape {}.".format(X.shape))
        cache = self.fill_cache(X)  # pylint: disable=E1128
        dlossds = self.dlossds(X, y, cache=cache)
        dlossdw = self.dlossdw()
        return self.gradient_backward(dlossds, dlossdw, X, inputs=inputs, cache=cache)

    def fit(self, X, y, optimizer=None, max_iter=100, early_th=None, verbose=False,
            lr=None, lr_schedule=None):
        """
        Fits a neuron.

        :param X: training set
        :param y: training labels
        :param optimizer: optimizer, by default, it is
            :class:`SGDOptimizer <mlstatpy.optim.sgd.SGDOptimizer>`.
        :param max_iter: number maximum of iterations
        :param early_th: early stopping threshold
        :param verbose: more verbose
        :param lr: to overwrite *learning_rate_init* if
            *optimizer* is None (unused otherwise)
        :param lr_schedule: to overwrite *lr_schedule* if
            *optimizer* is None (unused otherwise)
        :return: self
        """
        if optimizer is None:
            optimizer = SGDOptimizer(
                self.training_weights, learning_rate_init=lr or 0.002,
                lr_schedule=lr_schedule or 'invscaling')

        def fct_loss(coef, lx, ly, neuron=self):
            neuron.update_training_weights(coef, False)
            loss = neuron.loss(lx, ly)
            if loss.shape[0] > 1:
                return numpy.sum(loss)
            return loss

        def fct_grad(coef, lx, ly, i, neuron=self):
            neuron.update_training_weights(coef, False)
            return neuron.gradient(lx, ly).ravel()

        optimizer.train(
            X, y, fct_loss, fct_grad, max_iter=max_iter,
            early_th=early_th, verbose=verbose)

        self.update_training_weights(optimizer.coef, False)
        return self
