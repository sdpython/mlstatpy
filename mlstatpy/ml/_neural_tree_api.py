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
        raise NotImplementedError("This should be overwritten.")  # pragma: no cover

    def update_training_weights(self, grad, add=True):
        """
        Updates weights.

        :param grad: vector to add to the weights such as gradient
        :param add: addition or replace
        """
        raise NotImplementedError("This should be overwritten.")  # pragma: no cover

    def fill_cache(self, X):
        """
        Creates a cache with intermediate results.
        """
        return None  # pragma: no cover

    def loss(self, X, y, cache=None):
        """
        Computes the loss. Returns a float.
        """
        raise NotImplementedError("This should be overwritten.")  # pragma: no cover

    def dlossds(self, X, y, cache=None):
        """
        Computes the loss derivative due to prediction error.
        """
        raise NotImplementedError("This should be overwritten.")  # pragma: no cover

    def gradient_backward(self, graddx, X, inputs=False, cache=None):
        """
        Computes the gradient in X.

        :param graddx: existing gradient against the outputs
        :param X: computes the gradient in X
        :param inputs: if False, derivative against the coefficients,
            otherwise against the inputs.
        :param cache: cache intermediate results to avoid more computation
        :return: gradient
        """
        raise NotImplementedError("This should be overwritten.")  # pragma: no cover

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
                f"X must a vector of one dimension but has shape {X.shape}."
            )
        cache = self.fill_cache(X)  # pylint: disable=E1128
        dlossds = self.dlossds(X, y, cache=cache)
        return self.gradient_backward(dlossds, X, inputs=inputs, cache=cache)

    def fit(
        self,
        X,
        y,
        optimizer=None,
        max_iter=100,
        early_th=None,
        verbose=False,
        lr=None,
        lr_schedule=None,
        l1=0.0,
        l2=0.0,
        momentum=0.9,
    ):
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
        :param l1: L1 regularization if *optimizer* is None
            (unused otherwise)
        :param l2: L2 regularization if *optimizer* is None
            (unused otherwise)
        :param momentum: used if *optimizer* is None
        :return: self
        """
        if optimizer is None:
            optimizer = SGDOptimizer(
                self.training_weights,
                learning_rate_init=lr or 0.002,
                lr_schedule=lr_schedule or "invscaling",
                l1=l1,
                l2=l2,
                momentum=momentum,
            )

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
            X,
            y,
            fct_loss,
            fct_grad,
            max_iter=max_iter,
            early_th=early_th,
            verbose=verbose,
        )

        self.update_training_weights(optimizer.coef, False)
        return self
