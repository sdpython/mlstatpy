# -*- coding: utf-8 -*-
"""
@file
@brief Conversion from tree to neural network.
"""
import numpy
import numpy.random as rnd
from scipy.special import expit, softmax, rel_entr  # pylint: disable=E0611


class _TrainingAPI:
    """
    Declaration of function needed to train a model.
    """

    @property
    def training_weights(self):
        "Returns the weights."
        raise NotImplementedError(  # pragma: no cover
            "This should be overwritten.")

    def update_training_weights(self, grad):
        """
        Updates weights.

        :param grad: vector to add to the weights such as gradient
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
        Computes a loss. Returns a float.
        """
        raise NotImplementedError(  # pragma: no cover
            "This should be overwritten.")

    def dlossdx(self, X, y, cache=None):
        """
        Computes the loss derivative against the inputs.
        """
        raise NotImplementedError(  # pragma: no cover
            "This should be overwritten.")

    def dlossdw(self, X, y, cache=None):
        """
        Computes the loss derivative against the weights.
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
        cache = self.fill_cache(X)  # pylint: disable=E1128
        dlossdx = self.dlossdx(X, y, cache=cache)
        dlossdw = self.dlossdw(X, y, cache=cache)
        return self.gradient_backward(dlossdx, dlossdw, X, inputs=inputs, cache=cache)


class NeuralTreeNode(_TrainingAPI):
    """
    One node in a neural network.
    """

    @staticmethod
    def _relu(x):
        "Relu function."
        return x if x > 0 else 0

    @staticmethod
    def _drelu(x):
        "Derivative of the relu function."
        return 1 if x > 0 else 0

    @staticmethod
    def _dsigmoid(x):
        "Derivativ of the sigmoid function."
        y = expit(x)
        return y * (1 - y)

    @staticmethod
    def _dsoftmax(x):
        "Derivative of the softmax function."
        soft = softmax(x)
        ex = numpy.exp(x)
        exs = numpy.sum(ex)
        iexs = 1 / exs
        dinv = - soft * iexs
        niexs = numpy.full(ex.shape, iexs)
        diag = numpy.diag(niexs)
        return diag + dinv

    @staticmethod
    def get_activation_function(activation):
        """
        Returns the activation function.
        It returns a function *y=f(x)*.
        """
        if activation == 'softmax':
            return softmax
        if activation == 'softmax4':
            return lambda x: softmax(x * 4)
        if activation in {'logistic', 'expit', 'sigmoid'}:
            return expit
        if activation == 'sigmoid4':
            return lambda x: expit(x * 4)
        if activation == 'relu':
            return numpy.vectorize(NeuralTreeNode._relu)
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
            return numpy.vectorize(NeuralTreeNode._drelu)
        if activation == 'identity':
            return numpy.vectorize(lambda x: 1)
        raise ValueError(
            "Unknown activation gradient function '{}'.".format(activation))

    @staticmethod
    def get_activation_loss_function(activation):
        """
        Returns a default loss function based on the activation
        function. It returns a function *g=f'(w,x,y)*
        where *w* are the weights.
        """
        if activation in {'logistic', 'expit', 'sigmoid', 'sigmoid4'}:
            # regression + regularization
            return lambda w, x, y: (x - y) ** 2 + w @ w.T * 0.1
        if activation in {'softmax', 'softmax4'}:
            # classification
            return lambda w, x, y: rel_entr(x, y)
        if activation in {'identity', 'relu'}:
            # regression
            return lambda w, x, y: (x - y) ** 2
        raise ValueError(
            "Unknown activation function '{}'.".format(activation))

    @staticmethod
    def get_activation_dloss_function(activation):
        """
        Returns the derivative of the default loss function based
        on the activation function. It returns a function
        *df(w,x,y)/dw, df(w,x,y)/dx* where *w* are the weights.
        """
        if activation in {'logistic', 'expit', 'sigmoid', 'sigmoid4'}:
            # regression + regularization
            def dregrdx(w, x, y):
                return x * (x - y) * 2

            def dregrdw(w, x, y):
                return w * 2
            return dregrdx, dregrdw

        if activation in {'softmax', 'softmax4'}:
            # classification
            cst = numpy.finfo(numpy.float32).eps

            def dclsdx(w, x, y):
                return numpy.log(x + cst) - numpy.log(y + cst) + 1

            def dclsdw(w, x, y):
                return numpy.zeros(w.shape, dtype=w.dtype)
            return dclsdx, dclsdw

        if activation in {'identity', 'relu'}:
            # regression
            def dregdx(w, x, y):
                return x * (x - y) * 2

            def dregdw(w, x, y):
                return numpy.zeros(w.shape, dtype=w.dtype)
            return dregdx, dregdw
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
        self.loss_ = NeuralTreeNode.get_activation_loss_function(
            self.activation)
        self.dlossdx_, self.dlossdw_ = NeuralTreeNode.get_activation_dloss_function(
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
        return "%s(weights=%r, bias=%r, activation=%r)" % (
            self.__class__.__name__, self.coef[1:],
            self.coef[0], self.activation)

    def _predict(self, X):
        "Computes inputs of the activation function."
        if self.n_outputs == 1:
            return X @ self.coef[1:] + self.coef[0]
        return (X.reshape((1, -1)) @ self.coef[:, 1:].T + self.coef[:, 0]).ravel()

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

    @property
    def training_weights(self):
        "Returns the weights stored in the neuron."
        return self.coef.ravel()

    def update_training_weights(self, X):
        """
        Updates weights.

        :param grad: vector to add to the weights such as gradient
        """
        self.coef += X.reshape(self.coef.shape)

    def fill_cache(self, X):
        """
        Creates a cache with intermediate results.
        """
        return dict(sX=self._predict(X))

    def loss(self, X, y, cache=None):
        """
        Computes a loss. Returns a float.
        """
        if cache is not None and 'sX' in cache:
            self.loss_(cache['sX'], y)  # pylint: disable=E1120
        return self.loss_(self.predict(X), y)  # pylint: disable=E1120

    def dlossdx(self, X, y, cache=None):
        """
        Computes the loss derivative against the inputs.
        """
        if cache is not None and 'sX' in cache:
            self.dlossdx_(self.training_weights, cache['sX'], y)
        return self.dlossdx_(self.training_weights, self.predict(X), y)

    def dlossdw(self, X, y, cache=None):
        """
        Computes the loss derivative against the weights.
        """
        if cache is not None and 'sX' in cache:
            self.dlossdw_(self.training_weights, cache['sX'], y)
        return self.dlossdw_(self.training_weights, self.predict(X), y)

    def gradient_backward(self, graddx, graddw, X, inputs=False, cache=None):
        """
        Computes the gradients at point *X*.

        :param graddx: existing gradient against the inputs
        :param graddw: existing gradient against the weights
        :param X: computes the gradient in X
        :param inputs: if False, derivative against the coefficients,
            otherwise against the inputs.
        :return: gradient
        """
        if cache is None:
            cache = self.fill_cache(X)
        pred = cache['sX']
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
        return rgrad + graddw.reshape(rgrad.shape)


class NeuralTreeNet(_TrainingAPI):
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
        if len(node.input_weights.shape) == 1:
            if node.input_weights.shape[0] != len(inputs):
                raise RuntimeError(
                    "Dimension mismatch between weights [{}] and inputs [{}].".format(
                        node.input_weights.shape[0], len(inputs)))
            node.nodeid = len(self.nodes)
            self.nodes.append(node)
            attr = dict(inputs=numpy.array(inputs), output=self.size_)
            self.nodes_attr.append(attr)
            self.size_ += 1
        elif len(node.input_weights.shape) == 2:
            if node.input_weights.shape[1] != len(inputs):
                raise RuntimeError(
                    "Dimension mismatch between weights [{}] and inputs [{}].".format(
                        node.input_weights.shape[1], len(inputs)))
            node.nodeid = len(self.nodes)
            self.nodes.append(node)
            attr = dict(inputs=numpy.array(inputs),
                        output=list(range(self.size_, self.size_ + node.input_weights.shape[0])))
            self.nodes_attr.append(attr)
            self.size_ += node.input_weights.shape[0]
        else:
            raise RuntimeError(
                "Coefficients should have 1 or 2 dimension not {}.".format(node.input_weights.shape))

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
                                         activation='sigmoid4', tag="N%d-th" % i)
                root.append(node_th, feat_index)

                if i in predecessor:
                    pred = predecessor[i]
                    node1 = pred
                    node2 = node_th
                    attr1 = root[node1.nodeid][1]
                    attr2 = root[node2.nodeid][1]

                    coef = numpy.ones((2,), dtype=numpy.float64) * k
                    node_true = NeuralTreeNode(coef, bias=-k * 1.5,
                                               activation='sigmoid4',
                                               tag="N%d-T" % i)
                    root.append(node_true, [attr1['output'], attr2['output']])

                    coef = numpy.zeros((2,), dtype=numpy.float64)
                    coef[0] = k
                    coef[1] = -k
                    node_false = NeuralTreeNode(coef, bias=-k * 0.25,
                                                activation='sigmoid4',
                                                tag="N%d-F" % i)
                    root.append(node_false, [attr1['output'], attr2['output']])

                    predecessor[children_left[i]] = node_true
                    predecessor[children_right[i]] = node_false
                else:
                    coef = numpy.ones((1,), dtype=numpy.float64) * -1
                    node_false = NeuralTreeNode(
                        coef, bias=1, activation='identity', tag="N%d-F" % i)
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
            NeuralTreeNode(coef, bias=-k / 2,
                           activation='softmax4', tag="Nfinal"),
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
            stag = "" if self[i][0].tag is None else (self[i][0].tag + "\\n")
            bias = str(numpy.array(self[i][0].bias)).replace(" ", "&#92; ")
            if y is None:
                lab = lof % '{}{}id={} b={} s={}'.format(
                    stag, a, i, bias, self[i][0].n_outputs)
            else:
                yo = numpy.array(y[o])
                lab = lof % '{}{}id={} b={} s={}\ny={}'.format(
                    stag, a, i, bias, self[i][0].n_outputs, yo)
            rows.append('{} [label="{}"];'.format(
                lo, lab.replace("\n", "&#92;n")))
            for ii, inp in enumerate(self[i][1]['inputs']):
                if isinstance(o, int):
                    w = self[i][0].input_weights[ii]
                    if w == 0:
                        c = ', color=grey, fontcolor=grey'
                    elif w < 0:
                        c = ', color=red, fontcolor=red'
                    else:
                        c = ', color=blue, fontcolor=blue'
                    rows.append(
                        '{} -> {} [label="{}"{}];'.format(inp, o, w, c))
                    continue

                w = self[i][0].input_weights[:, ii]
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
    def training_weights(self):
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

    def update_training_weights(self, X):
        """
        Updates weights.

        :param grad: vector to add to the weights such as gradient
        """
        pos = 0
        for n in self.nodes:
            s = n.coef.size
            n.coef += X[pos: pos + s].reshape(n.coef.shape)
            pos += s

    def fill_cache(self, X):
        """
        Creates a cache with intermediate results.
        """
        raise NotImplementedError(  # pragma: no cover
            "This should be overwritten.")

    def loss(self, X, y, cache=None):
        """
        Computes a loss. Returns a float.
        """
        raise NotImplementedError(  # pragma: no cover
            "This should be overwritten.")

    def dlossdx(self, X, y, cache=None):
        """
        Computes the loss derivative against the inputs.
        """
        raise NotImplementedError(  # pragma: no cover
            "This should be overwritten.")

    def dlossdw(self, X, y, cache=None):
        """
        Computes the loss derivative against the weights.
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
