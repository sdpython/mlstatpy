# -*- coding: utf-8 -*-
"""
@file
@brief Conversion from tree to neural network.
"""
import numpy
import numpy.random as rnd
from scipy.special import expit, softmax  # pylint: disable=E0611


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
        if activation == 'sigmoid4':
            return lambda x: expit(x * 4)
        if activation == 'softmax':
            return softmax
        if activation == 'softmax4':
            return lambda x: softmax(x * 4)
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
        if isinstance(weights, int):
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
        self.activation_ = NeuralTreeNode.get_activation_function(activation)
        self.nodeid = nodeid

    @property
    def weights(self):
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
            'nodeid': self.nodeid, 'n_outputs': self.n_outputs}

    def __setstate__(self, state):
        "usual"
        self.coef = state['coef']
        self.activation = state['activation']
        self.nodeid = state['nodeid']
        self.n_outputs = state['n_outputs']
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
        if self.n_outputs == 1:
            return self.activation_(X @ self.coef[1:] + self.coef[0])
        return self.activation_(
            (X.reshape((1, -1)) @ self.coef[:, 1:].T + self.coef[:, 0]).ravel())

    @property
    def ndim(self):
        "Returns the input dimension."
        return self.coef.shape[0] - 1


class TrainingAPI:
    """
    Declaration of function needed to train a model.
    """

    @property
    def weights(self):
        "Returns the weights."
        raise NotImplementedError(  # pragma: no cover
            "This should be overwritten.")

    def update_weights(self, grad):
        """
        Updates weights.

        :param grad: vector to add to the weights such as gradient
        """
        raise NotImplementedError(  # pragma: no cover
            "This should be overwritten.")

    def gradient(self, X):
        """
        Computes the gradient in X.

        :param X: computes the gradient in X
        :return: gradient
        """
        raise NotImplementedError(  # pragma: no cover
            "This should be overwritten.")


class NeuralTreeNet(TrainingAPI):
    """
    Node ensemble.
    """

    def __init__(self, dim, empty=True):
        """
        @param      dim     space dimension
        @param      empty   empty network, other adds an identity node
        """
        self.dim = dim
        if empty:
            self.nodes = []
            self.nodes_attr = []
        else:
            self.nodes = [
                NeuralTreeNode(
                    numpy.ones((dim,), dtype=numpy.float64),
                    bias=numpy.float64(0.),
                    activation='identity', nodeid=0)]
            self.nodes_attr = [dict(inputs=numpy.arange(0, dim), output=dim)]
        self._update_members()

    def _update_members(self):
        if len(self.nodes_attr) == 0:
            self.size_ = self.dim
        else:
            self.size_ = max(d['output'] for d in self.nodes_attr) + 1

    def __repr__(self):
        "usual"
        return "%s(%d)" % (self.__class__.__name__, self.dim)

    def clear(self):
        "Clear all nodes"
        del self.nodes[:]
        del self.nodes_attr[:]
        self._update_members()

    def append(self, node, inputs):
        """
        Appends a node into the graph.

        @param      node        node to add
        @param      inputs      index of input nodes
        """
        if len(node.weights.shape) == 1:
            if node.weights.shape[0] != len(inputs):
                raise RuntimeError(
                    "Dimension mismatch between weights [{}] and inputs [{}].".format(
                        node.weights.shape[0], len(inputs)))
            node.nodeid = len(self.nodes)
            self.nodes.append(node)
            attr = dict(inputs=numpy.array(inputs), output=self.size_)
            self.nodes_attr.append(attr)
            self.size_ += 1
        elif len(node.weights.shape) == 2:
            if node.weights.shape[1] != len(inputs):
                raise RuntimeError(
                    "Dimension mismatch between weights [{}] and inputs [{}].".format(
                        node.weights.shape[1], len(inputs)))
            node.nodeid = len(self.nodes)
            self.nodes.append(node)
            attr = dict(inputs=numpy.array(inputs),
                        output=list(range(self.size_, self.size_ + node.weights.shape[0])))
            self.nodes_attr.append(attr)
            self.size_ += node.weights.shape[0]
        else:
            raise RuntimeError(
                "Coefficients should have 1 or 2 dimension not {}.".format(node.weights.shape))

    def __getitem__(self, i):
        "Retrieves node and attributes for node i."
        return self.nodes[i], self.nodes_attr[i]

    def __len__(self):
        "Returns the number of nodes"
        return len(self.nodes)

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

        n_nodes = tree.tree_.node_count
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        value = tree.tree_.value.reshape((-1, 2))
        output_class = (value[:, 1] > value[:, 0]).astype(numpy.int64)
        max_features_ = tree.max_features_

        root = NeuralTreeNet(tree.max_features_, empty=True)
        feat_index = numpy.arange(0, max_features_)
        predecessor = {}
        outputs = {i: [] for i in range(0, tree.n_classes_)}
        for i in range(n_nodes):

            if children_left[i] != children_right[i]:
                # node with a threshold
                # right side
                coef = numpy.zeros((max_features_,), dtype=numpy.float64)
                coef[feature[i]] = -k
                node_th = NeuralTreeNode(coef, bias=k * threshold[i],
                                         activation='sigmoid4')
                root.append(node_th, feat_index)

                if i in predecessor:
                    pred = predecessor[i]
                    node1 = pred
                    node2 = node_th
                    attr1 = root[node1.nodeid][1]
                    attr2 = root[node2.nodeid][1]

                    coef = numpy.ones((2,), dtype=numpy.float64) * k
                    node_true = NeuralTreeNode(coef, bias=-k * 1.5,
                                               activation='sigmoid4')
                    root.append(node_true, [attr1['output'], attr2['output']])

                    coef = numpy.zeros((2,), dtype=numpy.float64)
                    coef[0] = k
                    coef[1] = -k
                    node_false = NeuralTreeNode(coef, bias=-k * 0.25,
                                                activation='sigmoid4')
                    root.append(node_false, [attr1['output'], attr2['output']])

                    predecessor[children_left[i]] = node_true
                    predecessor[children_right[i]] = node_false
                else:
                    coef = numpy.ones((1,), dtype=numpy.float64) * -1
                    node_false = NeuralTreeNode(
                        coef, bias=1, activation='identity')
                    attr = root[node_th.nodeid][1]
                    root.append(node_false, [attr['output']])

                    predecessor[children_left[i]] = node_th
                    predecessor[children_right[i]] = node_false

            elif i in predecessor:
                # leave
                outputs[output_class[i]].append(predecessor[i])

        # final node
        output = []
        index = [0]
        nb = []
        for i in range(0, tree.n_classes_):
            output.extend(outputs[i])
            nb.append(len(outputs[i]))
            index.append(len(outputs[i]) + index[-1])
        coef = numpy.zeros((len(nb), len(output)), dtype=numpy.float64)
        for i in range(0, tree.n_classes_):
            coef[i, index[i]:index[i + 1]] = k
        feat = [root[n.nodeid][1]['output'] for n in output]
        root.append(
            NeuralTreeNode(coef, bias=-k / 2, activation='softmax4'),
            feat)

        # final
        return root

    def to_dot(self, X=None):
        """
        Exports the neural network into :epkg:`dot`.

        @param  X   input as an example
        """
        y = None
        if X is not None:
            y = self.predict(X)
        rows = ['digraph Tree {',
                "node [shape=box, fontsize=10];",
                "edge [fontsize=8];"]
        for i in range(self.dim):
            if y is None:
                rows.append('{0} [label="X[{0}]"];'.format(i))
            else:
                rows.append(
                    '{0} [label="X[{0}]=\\n{1:1.2f}"];'.format(i, X[i]))

        labels = {}

        for i in range(0, len(self)):  # pylint: disable=C0200
            o = self[i][1]['output']
            if isinstance(o, int):
                lo = str(o)
                labels[o] = lo
                lof = "%s"
            else:
                lo = "s" + 'a'.join(map(str, o))
                for oo in o:
                    labels[oo] = '{}:f{}'.format(lo, oo)
                los = "|".join("<f{0}> {0}".format(oo) for oo in o)
                lof = "%s&#92;n" + los

            a = "a={}\n".format(self[i][0].activation)
            bias = str(numpy.array(self[i][0].bias)).replace(" ", "&#92; ")
            if y is None:
                lab = lof % '{}id={} b={} s={}'.format(
                    a, i, bias, self[i][0].n_outputs)
            else:
                yo = numpy.array(y[o])
                lab = lof % '{}id={} b={} s={}\ny={}'.format(
                    a, i, bias, self[i][0].n_outputs, yo)
            rows.append('{} [label="{}"];'.format(
                lo, lab.replace("\n", "&#92;n")))
            for ii, inp in enumerate(self[i][1]['inputs']):
                if isinstance(o, int):
                    w = self[i][0].weights[ii]
                    if w == 0:
                        c = ', color=grey, fontcolor=grey'
                    elif w < 0:
                        c = ', color=red, fontcolor=red'
                    else:
                        c = ', color=blue, fontcolor=blue'
                    rows.append(
                        '{} -> {} [label="{}"{}];'.format(inp, o, w, c))
                    continue

                w = self[i][0].weights[:, ii]
                for oi, oo in enumerate(o):
                    if w[oi] == 0:
                        c = ', color=grey, fontcolor=grey'
                    elif w[oi] < 0:
                        c = ', color=red, fontcolor=red'
                    else:
                        c = ', color=blue, fontcolor=blue'
                    rows.append('{} -> {} [label="{}|{}"{}];'.format(
                        inp, labels[oo], oi, w[oi], c))

        rows.append('}')
        return '\n'.join(rows)

    @property
    def shape(self):
        "Returns the shape of the coefficients."
        return (sum(n.coef.size for n in self.nodes), )

    @property
    def weights(self):
        "Returns the weights."
        sh = self.shape
        res = numpy.empty(sh[0], dtype=numpy.float64)
        pos = 0
        for n in self.nodes:
            s = n.coef.size
            res[pos: pos +
                s] = n.coef if len(n.coef.shape) == 1 else n.coef.ravel()
            pos += s
        return res

    def update_weights(self, X):
        """
        Updates weights.

        :param grad: vector to add to the weights such as gradient
        """
        pos = 0
        for n in self.nodes:
            s = n.coef.size
            n.coef += X[pos: pos + s].reshape(n.coef.shape)
            pos += s

    def gradient(self, X):
        """
        Computes the gradient in X.

        :param X: computes the gradient in X
        :return: gradient
        """
        raise NotImplementedError(  # pragma: no cover
            "This should be overwritten.")
