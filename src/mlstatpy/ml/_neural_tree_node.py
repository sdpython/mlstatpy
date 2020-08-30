# -*- coding: utf-8 -*-
"""
@file
@brief Conversion from tree to neural network.
"""
import numpy
import numpy.random as rnd
from scipy.special import expit, softmax, kl_div as kl_fct  # pylint: disable=E0611
from ._neural_tree_api import _TrainingAPI


class NeuralTreeNode(_TrainingAPI):
    """
    One node in a neural network.
    """

    @staticmethod
    def _relu(x):
        "Relu function."
        return numpy.maximum(x, 0)

    @staticmethod
    def _leakyrelu(x):
        "Leaky Relu function."
        return numpy.maximum(x, 0) + numpy.minimum(x, 0) * 0.01

    @staticmethod
    def _drelu(x):
        "Derivative of the Relu function."
        res = numpy.ones(x.shape, dtype=x.dtype)
        res[x < 0] = 0.
        return res

    @staticmethod
    def _dleakyrelu(x):
        "Derivative of the Leaky Relu function."
        res = numpy.ones(x.shape, dtype=x.dtype)
        res[x < 0] = 0.01
        return res

    @staticmethod
    def _dsigmoid(x):
        "Derivativ of the sigmoid function."
        y = expit(x)
        return y * (1 - y)

    @staticmethod
    def _softmax(x):
        "Derivative of the softmax function."
        if len(x.shape) == 2:
            return softmax(x, axis=1)
        return softmax(x)

    @staticmethod
    def _dsoftmax(x):
        "Derivative of the softmax function."
        soft = softmax(x)
        grad = - soft @ soft.T
        diag = numpy.diag(soft)
        return diag + grad

    @staticmethod
    def get_activation_function(activation):
        """
        Returns the activation function.
        It returns a function *y=f(x)*.
        """
        if activation == 'softmax':
            return NeuralTreeNode._softmax
        if activation == 'softmax4':
            return lambda x: NeuralTreeNode._softmax(x * 4)
        if activation in {'logistic', 'expit', 'sigmoid'}:
            return expit
        if activation == 'sigmoid4':
            return lambda x: expit(x * 4)
        if activation == 'relu':
            return NeuralTreeNode._relu
        if activation == 'leakyrelu':
            return NeuralTreeNode._leakyrelu
        if activation == 'identity':
            return lambda x: x
        raise ValueError(
            "Unknown activation function '{}'.".format(activation))

    @staticmethod
    def get_activation_gradient_function(activation):
        """
        Returns the activation function.
        It returns a function *y=f'(x)*.
        About the sigmoid:

        .. math::

            \\begin{array}{l}
            f(x) &=& \frac{1}{1 + e^{-x}} \\\\
            f'(x) &=& \frac{e^{-x}}{(1 + e^{-x})^2} = f(x)(1-f(x))
            \\end{array}}
        """
        if activation == 'softmax':
            return NeuralTreeNode._dsoftmax
        if activation == 'softmax4':
            return lambda x: NeuralTreeNode._dsoftmax(x) * 4
        if activation in {'logistic', 'expit', 'sigmoid'}:
            return NeuralTreeNode._dsigmoid
        if activation == 'sigmoid4':
            return lambda x: NeuralTreeNode._dsigmoid(x) * 4
        if activation == 'relu':
            return NeuralTreeNode._drelu
        if activation == 'leakyrelu':
            return NeuralTreeNode._dleakyrelu
        if activation == 'identity':
            return lambda x: numpy.ones(x.shape, dtype=x.dtype)
        raise ValueError(
            "Unknown activation gradient function '{}'.".format(activation))

    @staticmethod
    def get_activation_loss_function(activation):
        """
        Returns a default loss function based on the activation
        function. It returns two functions *g=loss(x,y)*.
        """
        if activation in {'logistic', 'expit', 'sigmoid', 'sigmoid4'}:
            # regression + regularization
            return lambda x, y: (x - y) ** 2
        if activation in {'softmax', 'softmax4'}:
            cst = numpy.finfo(numpy.float32).eps

            # classification
            def kl_fct2(x, y):
                return kl_fct(x + cst, y + cst)
            return kl_fct2
        if activation in {'identity', 'relu', 'leakyrelu'}:
            # regression
            return lambda x, y: (x - y) ** 2
        raise ValueError(
            "Unknown activation function '{}'.".format(activation))

    @staticmethod
    def get_activation_dloss_function(activation):
        """
        Returns the derivative of the default loss function based
        on the activation function. It returns a function
        *df(x,y)/dw, df(w)/dw* where *w* are the weights.
        """
        if activation in {'logistic', 'expit', 'sigmoid', 'sigmoid4'}:
            # regression + regularization
            def dregrdx(x, y):
                return (x - y) * 2

            return dregrdx

        if activation in {'softmax', 'softmax4'}:
            # classification
            cst = numpy.finfo(numpy.float32).eps

            def dclsdx(x, y):
                return numpy.log(x + cst) - numpy.log(y + cst)

            return dclsdx

        if activation in {'identity', 'relu', 'leakyrelu'}:
            # regression
            def dregdx(x, y):
                return (x - y) * 2

            return dregdx
        raise ValueError(
            "Unknown activation function '{}'.".format(activation))

    def __init__(self, weights, bias=None, activation='sigmoid', nodeid=-1,
                 tag=None):
        """
        @param      weights     weights
        @param      bias        bias, if None, draws a random number
        @param      activation  activation function
        @param      nodeid      node id
        @param      tag         unused but to add information
                                on how this node was created
        """
        self.tag = tag
        if isinstance(weights, int):
            if activation.startswith('softmax'):
                weights = rnd.randn(2, weights)
            else:
                weights = rnd.randn(weights)
        if isinstance(weights, list):
            weights = numpy.array(weights)

        if len(weights.shape) == 1:
            self.n_outputs = 1
            if bias is None:
                bias = rnd.randn()
            self.coef = numpy.empty(len(weights) + 1)
            self.coef[1:] = weights
            self.coef[0] = bias

        elif len(weights.shape) == 2:
            self.n_outputs = weights.shape[0]
            if self.n_outputs == 1:
                raise RuntimeError(  # pragma: no cover
                    "Unexpected unsqueezed weights shape: {}".format(weights.shape))
            if bias is None:
                bias = rnd.randn(self.n_outputs)
            shape = list(weights.shape)
            shape[1] += 1
            self.coef = numpy.empty(shape)
            self.coef[:, 1:] = weights
            self.coef[:, 0] = bias
        else:
            raise RuntimeError(  # pragma: no cover
                "Unexpected weights shape: {}".format(weights.shape))

        self.activation = activation
        self.nodeid = nodeid
        self._set_fcts()

    def _set_fcts(self):
        self.activation_ = NeuralTreeNode.get_activation_function(
            self.activation)
        self.gradient_ = NeuralTreeNode.get_activation_gradient_function(
            self.activation)
        self.losss_ = NeuralTreeNode.get_activation_loss_function(
            self.activation)
        self.dlossds_ = NeuralTreeNode.get_activation_dloss_function(
            self.activation)

    @property
    def input_weights(self):
        "Returns the weights."
        if self.n_outputs == 1:
            return self.coef[1:]
        return self.coef[:, 1:]

    @property
    def bias(self):
        "Returns the weights."
        if self.n_outputs == 1:
            return self.coef[0]
        return self.coef[:, 0]

    def __getstate__(self):
        "usual"
        return {
            'coef': self.coef, 'activation': self.activation,
            'nodeid': self.nodeid, 'n_outputs': self.n_outputs,
            'tag': self.tag}

    def __setstate__(self, state):
        "usual"
        self.coef = state['coef']
        self.activation = state['activation']
        self.nodeid = state['nodeid']
        self.n_outputs = state['n_outputs']
        self.tag = state['tag']
        self._set_fcts()

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
        if len(self.coef.shape) == 1:
            return "%s(weights=%r, bias=%r, activation=%r)" % (
                self.__class__.__name__, self.coef[1:],
                self.coef[0], self.activation)
        return "%s(weights=%r, bias=%r, activation=%r)" % (
            self.__class__.__name__, self.coef[:, 1:],
            self.coef[:, 0], self.activation)

    def _predict(self, X):
        "Computes inputs of the activation function."
        if self.n_outputs == 1:
            return X @ self.coef[1:] + self.coef[0]
        return (X.reshape((1, -1)) @ self.coef[:, 1:].T + self.coef[:, 0]).ravel()

    def predict(self, X):
        "Computes neuron outputs."
        if self.n_outputs == 1:
            return self.activation_(X @ self.coef[1:] + self.coef[0])
        if len(X.shape) == 2:
            return self.activation_(
                (X @ self.coef[:, 1:].T + self.coef[:, 0]))
        return self.activation_(
            (X.reshape((1, -1)) @ self.coef[:, 1:].T + self.coef[:, 0]).ravel())

    @property
    def ndim(self):
        "Returns the input dimension."
        return self.coef.shape[0] - 1

    @property
    def training_weights(self):
        "Returns the weights stored in the neuron."
        return self.coef.ravel()

    def update_training_weights(self, X, add=True):
        """
        Updates weights.

        :param grad: vector to add to the weights such as gradient
        :param add: addition or replace
        """
        if add:
            self.coef += X.reshape(self.coef.shape)
        else:
            numpy.copyto(self.coef, X.reshape(self.coef.shape))

    def fill_cache(self, X):
        """
        Creates a cache with intermediate results.
        ``lX`` is the results before the activation function,
        ``aX`` is the results after the activation function, the prediction.
        """
        cache = dict(lX=self._predict(X))
        cache['aX'] = self.activation_(cache['lX'])
        return cache

    def _common_loss_dloss(self, X, y, cache=None):
        """
        Common beginning to methods *loss*, *dlossds*,
        *dlossdw*.
        """
        if cache is not None and 'aX' in cache:
            act = cache['aX']
        else:
            act = self.predict(X)
        return act

    def loss(self, X, y, cache=None):
        """
        Computes the loss. Returns a float.
        """
        act = self._common_loss_dloss(X, y, cache=cache)
        if len(X.shape) == 1:
            return self.losss_(act, y)  # pylint: disable=E1120
        return self.losss_(act, y)  # pylint: disable=E1120

    def dlossds(self, X, y, cache=None):
        """
        Computes the loss derivative due to prediction error.
        """
        act = self._common_loss_dloss(X, y, cache=cache)
        return self.dlossds_(act, y)

    def gradient_backward(self, graddx, X, inputs=False, cache=None):
        """
        Computes the gradients at point *X*.

        :param graddx: existing gradient against the inputs
        :param X: computes the gradient in X
        :param inputs: if False, derivative against the coefficients,
            otherwise against the inputs.
        :param cache: cache intermediate results
        :return: gradient
        """
        if cache is None:
            cache = self.fill_cache(X)

        pred = cache['aX']
        ga = self.gradient_(pred)
        if len(ga.shape) == 2:
            f = graddx @ ga
        else:
            f = graddx * ga

        if inputs:
            if len(self.coef.shape) == 1:
                rgrad = numpy.empty(X.shape)
                rgrad[:] = self.coef[1:]
                rgrad *= f
            else:
                rgrad = numpy.sum(
                    self.coef[:, 1:] * f.reshape((-1, 1)), axis=0)
            return rgrad

        rgrad = numpy.empty(self.coef.shape)
        if len(self.coef.shape) == 1:
            rgrad[0] = 1
            rgrad[1:] = X
            rgrad *= f
        else:
            rgrad[:, 0] = 1
            rgrad[:, 1:] = X
            rgrad *= f.reshape((-1, 1))
        return rgrad
