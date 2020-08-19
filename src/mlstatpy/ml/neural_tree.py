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
        if activation in {'logistic', 'expit', 'sigmoid'}:
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
        if node.weights.shape[0] != len(inputs):
            raise RuntimeError(
                "Dimension mismatch between weights [{}] and inputs [{}].".format(
                    node.weights.shape[0], len(inputs)))
        node.nodeid = len(self.nodes)
        self.nodes.append(node)
        attr = dict(inputs=numpy.array(inputs), output=self.size_)
        self.nodes_attr.append(attr)
        self.size_ += 1

    def __getitem__(self, i):
        "Retrieves node and attributes for node i."
        return self.nodes[i], self.nodes_attr[i]

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

    @staticmethod
    def create_from_tree(tree, k=1.):
        """
        Creates a @see cl NeuralTreeNet instance from a
        :epkg:`DecisionTreeClassifier`

        @param  tree    :epkg:`DecisionTreeClassifier`
        @param  k       slant of the sigmoïd
        @return         @see cl NeuralTreeNet

        The function only works for binary problems.
        """
        if tree.n_classes_ > 2:
            raise RuntimeError(
                "The function only support binary classification problem.")
        root = NeuralTreeNet(tree.max_features_)
        index = {}

        n_nodes = tree.tree_.node_count
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        max_features_ = tree.max_features_

        feat_index = numpy.arange(0, max_features_)
        predecessor = {}
        output = []
        for i in range(n_nodes):
            # right side
            coef = numpy.zeros((max_features_,), dtype=numpy.float64)
            coef[feature[i]] = k
            node = NeuralTreeNode(coef, bias=-k * threshold[i])
            root.append(node, feat_index)
            predecessor[children_left[i]] = (i, 0)
            predecessor[children_right[i]] = (i, 1)
            index[i] = node

            if i in predecessor:
                pred, side = predecessor[i]
                if pred not in index:
                    raise RuntimeError("Unxpected predecessor %r." % pred)

                node1 = index[pred]
                node2 = node
                attr1 = root[node1.nodeid][1]
                attr2 = root[node2.nodeid][1]

                if side == 1:
                    coef = numpy.ones((2,), dtype=numpy.float64) * k
                    node = NeuralTreeNode(coef, bias=-k / 2)
                else:
                    coef = numpy.zeros((2,), dtype=numpy.float64)
                    coef[0] = k
                    coef[1] = -k
                    node = NeuralTreeNode(coef, bias=k / 2)
                root.append(node, [attr1['output'], attr2['output']])
            elif children_left[i] != children_right[i]:
                output.append(node)

        # final node
        coef = numpy.ones(
            (len(output), ), dtype=numpy.float64) / len(output) * k
        feat = [root[n.nodeid][1]['output'] for n in output]
        root.append(NeuralTreeNode(coef, bias=-k * threshold[i]), feat)

        # final
        return root
