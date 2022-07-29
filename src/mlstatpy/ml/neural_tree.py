# -*- coding: utf-8 -*-
"""
@file
@brief Conversion from tree to neural network.
"""
from io import BytesIO
import pickle
import numpy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import BaseDecisionTree
from ._neural_tree_api import _TrainingAPI
from ._neural_tree_node import NeuralTreeNode


def label_class_to_softmax_output(y_label):
    """
    Converts a binary class label into a matrix
    with two columns of probabilities.

    .. runpython::
        :showcode:

        import numpy
        from mlstatpy.ml.neural_tree import label_class_to_softmax_output

        y_label = numpy.array([0, 1, 0, 0])
        soft_y = label_class_to_softmax_output(y_label)
        print(soft_y)
    """
    if len(y_label.shape) != 1:
        raise ValueError(
            f"y_label must be a vector but has shape {y_label.shape}.")
    y = numpy.empty((y_label.shape[0], 2), dtype=numpy.float64)
    y[:, 0] = (y_label < 0.5).astype(numpy.float64)
    y[:, 1] = 1 - y[:, 0]
    return y


class NeuralTreeNet(_TrainingAPI):
    """
    Node ensemble.

    .. runpython::
        :showcode:

        import numpy
        from mlstatpy.ml.neural_tree import NeuralTreeNode, NeuralTreeNet

        w1 = numpy.array([-0.5, 0.8, -0.6])

        neu = NeuralTreeNode(w1[1:], bias=w1[0], activation='sigmoid')
        net = NeuralTreeNet(2, empty=True)
        net.append(neu, numpy.arange(2))

        ide = NeuralTreeNode(numpy.array([1.]),
                             bias=numpy.array([0.]),
                             activation='identity')

        net.append(ide, numpy.arange(2, 3))

        X = numpy.abs(numpy.random.randn(10, 2))
        pred = net.predict(X)
        print(pred)
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
            self.nodes_attr = [dict(inputs=numpy.arange(0, dim), output=dim,
                                    coef_size=self.nodes[0].coef.size,
                                    first_coef=0)]
        self._update_members()

    def copy(self):
        st = BytesIO()
        pickle.dump(self, st)
        cop = BytesIO(st.getvalue())
        return pickle.load(cop)

    def _update_members(self, node=None, attr=None):
        "Updates internal members."
        if node is None or attr is None:
            if len(self.nodes_attr) == 0:
                self.size_ = self.dim
            else:
                self.size_ = max(d['output'] for d in self.nodes_attr) + 1
            self.output_to_node_ = {}
            self.input_to_node_ = {}
            for node2, attr2 in zip(self.nodes, self.nodes_attr):
                if isinstance(attr2['output'], list):
                    for o in attr2['output']:
                        self.output_to_node_[o] = node2, attr2
                else:
                    self.output_to_node_[attr2['output']] = node2, attr2
                for i in attr2['inputs']:
                    self.input_to_node_[i] = node2, attr2
        else:
            if len(node.input_weights.shape) == 1:
                self.size_ += 1
            else:
                self.size_ += node.input_weights.shape[0]
            if isinstance(attr['output'], list):
                for o in attr['output']:
                    self.output_to_node_[o] = node, attr
            else:
                self.output_to_node_[attr['output']] = node, attr
            for i in attr['inputs']:
                self.input_to_node_[i] = node, attr

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
                    f"Dimension mismatch between weights "
                    f"[{node.input_weights.shape[0]}] "
                    f"and inputs [{len(inputs)}].")
            node.nodeid = len(self.nodes)
            self.nodes.append(node)
            first_coef = (
                0 if len(self.nodes_attr) == 0 else
                self.nodes_attr[-1]['first_coef'] + self.nodes_attr[-1]['coef_size'])
            attr = dict(inputs=numpy.array(inputs), output=self.size_,
                        coef_size=node.coef.size, first_coef=first_coef)
            self.nodes_attr.append(attr)
        elif len(node.input_weights.shape) == 2:
            if node.input_weights.shape[1] != len(inputs):
                raise RuntimeError(  # pragma: no cover
                    f"Dimension mismatch between weights "
                    f"[{node.input_weights.shape[1]}] "
                    f"and inputs [{len(inputs)}], tag={node.tag!r}, "
                    f"node={node!r}.")
            node.nodeid = len(self.nodes)
            self.nodes.append(node)
            first_coef = (
                0 if len(self.nodes_attr) == 0 else
                self.nodes_attr[-1]['first_coef'] + self.nodes_attr[-1]['coef_size'])
            attr = dict(inputs=numpy.array(inputs),
                        output=list(range(self.size_, self.size_ +
                                          node.input_weights.shape[0])),
                        coef_size=node.coef.size, first_coef=first_coef)
            self.nodes_attr.append(attr)
        else:
            raise RuntimeError(  # pragma: no cover
                f"Coefficients should have 1 or 2 dimension not "
                f"{node.input_weights.shape}.")
        self._update_members(node, attr)

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
    def create_from_tree(tree, k=1., arch='one'):
        """
        Creates a @see cl NeuralTreeNet instance from a
        :epkg:`DecisionTreeClassifier`

        @param  tree    :epkg:`DecisionTreeClassifier`
        @param  k       slant of the sigmoïd
        @param  arch    architecture, see below
        @return         @see cl NeuralTreeNet

        The function only works for binary problems.
        Available architecture:
        * `'one'`: the method adds nodes with one output, there
          is no soecific definition of layers,
        * `'compact'`: the adds two nodes, the first computes
          the threshold, the second one computes the leaves
          output, a final node merges all outputs into one

        See notebook :ref:`neuraltreerst` for examples.
        """
        if arch == 'one':
            return NeuralTreeNet._create_from_tree_one(tree, k)
        if arch == 'compact':
            return NeuralTreeNet._create_from_tree_compact(tree, k)
        raise ValueError(f"Unknown arch value '{arch}'.")

    @staticmethod
    def _create_from_tree_one(tree, k=1.):
        "Implements strategy 'one'. See @see meth create_from_tree."

        if not isinstance(tree, BaseDecisionTree):
            raise TypeError(  # pragma: no cover
                f"Only decision tree as supported not {type(tree)!r}.")
        if tree.n_classes_ > 2:
            raise RuntimeError(  # pragma: no cover
                "The function only supports binary classification problem.")

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
            NeuralTreeNode(coef, bias=(-k / 2) * len(feat),
                           activation='softmax4', tag="Nfinal"),
            feat)

        # final
        return root

    @staticmethod
    def _create_from_tree_compact(tree, k=1.):
        "Implements strategy 'compact'. See @see meth create_from_tree."
        if not isinstance(tree, BaseDecisionTree):
            raise TypeError(  # pragma: no cover
                f"Only decision tree as supported not {type(tree)!r}.")
        if isinstance(tree, ClassifierMixin):
            is_classifier = True
            if tree.n_classes_ > 2:
                raise RuntimeError(  # pragma: no cover
                    "The function only supports binary classification problem.")
        else:
            is_classifier = False
            if tree.n_outputs_ != 1:
                raise RuntimeError(  # pragma: no cover
                    "The function only supports single regression problem.")

        n_nodes = tree.tree_.node_count
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        if is_classifier:
            value = tree.tree_.value.reshape((-1, 2))
            output_class = (value[:, 1] > value[:, 0]).astype(numpy.int64)
        else:
            output_value = tree.tree_.value.ravel()
        max_features_ = tree.max_features_
        feat_index = numpy.arange(0, max_features_)

        root = NeuralTreeNet(tree.max_features_, empty=True)
        coef1 = []
        bias1 = []
        parents = {}
        rows = {}

        # first pass: threshold

        for i in range(n_nodes):
            if children_left[i] == children_right[i]:
                # leaves
                continue
            rows[i] = len(coef1)
            parents[children_left[i]] = i
            parents[children_right[i]] = i
            coef = numpy.zeros((max_features_,), dtype=numpy.float64)
            coef[feature[i]] = -k
            coef1.append(coef)
            bias1.append(k * threshold[i])

        coef1 = numpy.vstack(coef1)
        if len(bias1) == 1:
            bias1 = bias1[0]
        node1 = NeuralTreeNode(
            coef1 if coef1.shape[0] > 1 else coef1[0], bias=bias1,
            activation='sigmoid4', tag="threshold")
        root.append(node1, feat_index)
        th_index = numpy.arange(max_features_, max_features_ + coef1.shape[0])

        # second pass: decision path
        coef2 = []
        bias2 = []
        output = []
        paths = []

        for i in range(n_nodes):
            if children_left[i] != children_right[i]:
                # not a leave
                continue

            path = []
            last = i
            if is_classifier:
                lr = "class", output_class[i]
                output.append(output_class[i])
            else:
                lr = "reg", output_value[i]
                output.append(output_value[i])
            while last is not None:
                path.append((last, lr))
                if last not in parents:
                    break
                par = parents[last]
                if children_right[par] == last:
                    lr = 'right'
                elif children_left[par] == last:
                    lr = 'left'
                else:
                    raise RuntimeError(  # pragma: no cover
                        "Inconsistent tree structure.")
                last = par

            coef = numpy.zeros((coef1.shape[0], ), dtype=numpy.float64)
            bias = 0.
            for ip, lr in path:
                if isinstance(lr, tuple):
                    lr, value = lr
                    if lr not in ('class', 'reg'):
                        raise RuntimeError(  # pragma: no cover
                            "algorithm issue")
                else:
                    r = rows[ip]
                    if lr == 'right':
                        coef[r] = k
                        bias -= k / 2
                    else:
                        coef[r] = -k
                        bias += k / 2
            coef2.append(coef)
            bias2.append(bias)
            paths.append(path)

        coef2 = numpy.vstack(coef2)
        if len(bias2) == 1:
            bias2 = bias2[0]
        node2 = NeuralTreeNode(
            coef2 if coef2.shape[0] > 1 else coef2[0], bias=bias2,
            activation='sigmoid4', tag="pathes")
        root.append(node2, th_index)

        # final node
        n_outputs = tree.n_classes_ if is_classifier else tree.n_outputs_
        
        index1 = max_features_ + coef1.shape[0]
        index2 = index1 + coef2.shape[0]
        findex = numpy.arange(index1, index2)

        if is_classifier:
            coef = numpy.zeros((n_outputs, coef2.shape[0]), dtype=numpy.float64)
            bias = numpy.zeros(n_outputs, dtype=numpy.float64)
            for i, cls in enumerate(output):
                coef[cls, i] = -k
                coef[1 - cls, i] = k
                bias[cls] += k / 2
                bias[1 - cls] += -k / 2
            root.append(
                NeuralTreeNode(coef, bias=bias,
                               activation='softmax4', tag="final"),
                findex)
        else:
            coef = numpy.array(output, dtype=numpy.float64)
            bias = numpy.zeros(n_outputs, dtype=numpy.float64)
            for i, reg in enumerate(output):
                coef[i] = reg
            root.append(
                NeuralTreeNode(coef, bias=bias,
                               activation='identity', tag="final"),
                findex)

        # end
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
                    labels[oo] = f'{lo}:f{oo}'
                los = "|".join("<f{0}> {0}".format(oo) for oo in o)
                lof = "%s&#92;n" + los

            a = f"a={self[i][0].activation}\n"
            stag = "" if self[i][0].tag is None else (self[i][0].tag + "\\n")
            bias = str(numpy.array(self[i][0].bias)).replace(" ", "&#92; ")
            if y is None:
                lab = lof % f'{stag}{a}id={i} b={bias} s={self[i][0].n_outputs}'
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
                        f'{inp} -> {o} [label="{w}"{c}];')
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
                        labels.get(inp, inp), labels[oo], oi, w[oi], c))

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
            res[pos: pos + s] = (
                n.coef if len(n.coef.shape) == 1 else n.coef.ravel())
            pos += s
        return res

    def update_training_weights(self, X, add=True):  # pylint: disable=W0237
        """
        Updates weights.

        :param grad: vector to add to the weights such as gradient
        :param add: addition or replace
        """
        pos = 0
        if add:
            for n in self.nodes:
                s = n.coef.size
                n.coef += X[pos: pos + s].reshape(n.coef.shape)
                pos += s
        else:
            for n in self.nodes:
                s = n.coef.size
                numpy.copyto(n.coef, X[pos: pos + s].reshape(n.coef.shape))
                pos += s

    def fill_cache(self, X):
        """
        Creates a cache with intermediate results.
        """
        big_cache = {}
        res = numpy.zeros((self.size_,), dtype=numpy.float64)
        res[:self.dim] = X
        for node, attr in zip(self.nodes, self.nodes_attr):
            cache = node.fill_cache(res[attr['inputs']])
            big_cache[node.nodeid] = cache
            res[attr['output']] = cache['aX']
        big_cache[-1] = res
        return big_cache

    def _get_output_node_attr(self, nb_last):
        """
        Retrieves the output nodes.
        *nb_last* is the number of expected outputs.
        """
        neurones = set(self.output_to_node_[i][0].nodeid
                       for i in range(self.size_ - nb_last, self.size_))
        if len(neurones) != 1:
            raise RuntimeError(  # pragma: no cover
                f"Only one output node is implemented not {len(neurones)}")
        return self.output_to_node_[self.size_ - 1]

    def _common_loss_dloss(self, X, y, cache=None):
        """
        Common beginning to methods *loss*, *dlossds*,
        *dlossdw*.
        """
        last = 1 if len(y.shape) <= 1 else y.shape[1]
        if cache is not None and -1 in cache:
            res = cache[-1]
        else:
            res = self.predict(X)
        if len(res.shape) == 2:
            pred = res[:, -last:]
        else:
            pred = res[-last:]
        last_node, last_attr = self._get_output_node_attr(last)
        return res, pred, last_node, last_attr

    def loss(self, X, y, cache=None):
        """
        Computes the loss due to prediction error. Returns a float.
        """
        res, _, last_node, last_attr = self._common_loss_dloss(
            X, y, cache=cache)
        if len(res.shape) <= 1:
            return last_node.loss(res[last_attr['inputs']], y)  # pylint: disable=E1120
        return last_node.loss(res[:, last_attr['inputs']], y)  # pylint: disable=E1120

    def dlossds(self, X, y, cache=None):
        """
        Computes the loss derivative against the inputs.
        """
        res, _, last_node, last_attr = self._common_loss_dloss(
            X, y, cache=cache)
        if len(res.shape) <= 1:
            return last_node.dlossds(res[last_attr['inputs']], y)  # pylint: disable=E1120
        return last_node.dlossds(res[:, last_attr['inputs']], y)  # pylint: disable=E1120

    def gradient_backward(self, graddx, X, inputs=False, cache=None):
        """
        Computes the gradient in X.

        :param graddx: existing gradient against the inputs
        :param X: computes the gradient in X
        :param inputs: if False, derivative against the coefficients,
            otherwise against the inputs.
        :param cache: cache intermediate results to avoid more computation
        :return: gradient
        """
        if cache is None:
            cache = self.fill_cache(X)
        shape = self.training_weights.shape
        pred = self.predict(X)

        whole_gradx = numpy.zeros(pred.shape, dtype=numpy.float64)
        whole_gradw = numpy.zeros(shape, dtype=numpy.float64)
        if len(graddx.shape) == 0:
            whole_gradx[-1] = graddx
        else:
            whole_gradx[-graddx.shape[0]:] = graddx

        for node, attr in zip(self.nodes[::-1], self.nodes_attr[::-1]):
            ch = cache[node.nodeid]

            node_graddx = whole_gradx[attr['output']]
            xi = pred[attr['inputs']]

            temp_gradw = node.gradient_backward(
                node_graddx, xi, inputs=False, cache=ch)
            temp_gradx = node.gradient_backward(
                node_graddx, xi, inputs=True, cache=ch)

            whole_gradw[attr['first_coef']:attr['first_coef'] +
                        attr['coef_size']] += temp_gradw.reshape((attr['coef_size'],))
            whole_gradx[attr['inputs']
                        ] += temp_gradx.reshape((len(attr['inputs']),))

        if inputs:
            return whole_gradx
        return whole_gradw


class NeuralTreeNetClassifier(ClassifierMixin, BaseEstimator):
    """
    Classifier following :epkg:`scikit-learn` API.

    :param estimator: instance of @see cl NeuralTreeNet.
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
    """

    def __init__(self, estimator,
                 optimizer=None, max_iter=100, early_th=None, verbose=False,
                 lr=None, lr_schedule=None, l1=0., l2=0., momentum=0.9):
        if not isinstance(estimator, NeuralTreeNet):
            raise ValueError(  # pragma: no cover
                f"estimator must be an instance of NeuralTreeNet not {type(estimator)!r}.")
        BaseEstimator.__init__(self)
        ClassifierMixin.__init__(self)
        self.estimator_ = estimator
        self.optimizer = None
        self.max_iter = max_iter
        self.early_th = early_th
        self.verbose = verbose
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.l1 = l1
        self.l2 = l2
        self.momentum = momentum

    def predict(self, X):
        """
        Returns the predicted classes.

        :param X: inputs
        :return: classes
        """
        probas = self.predict_proba(X)
        return numpy.argmax(probas, axis=1)

    def predict_proba(self, X):
        """
        Returns the classification probabilities.

        :param X: inputs
        :return: probabilities
        """
        return self.decision_function(X)[:, -2:]

    def decision_function(self, X):
        """
        Returns the classification probabilities.

        :param X: inputs
        :return: probabilities
        """
        return self.estimator_.predict(X)

    def fit(self, X, y, sample_weights=None):
        """
        Trains the estimator.

        :param X: input features
        :param y: expected classes (binary)
        :param sample_weights: sample weights
        :return: self
        """
        if sample_weights is not None:
            raise NotImplementedError(  # pragma: no cover
                "sample_weights is not supported yet.")
        ny = label_class_to_softmax_output(y)
        self.estimator_.fit(X, ny, optimizer=self.optimizer, max_iter=self.max_iter,
                            early_th=self.early_th, verbose=self.verbose,
                            lr=self.lr, lr_schedule=self.lr_schedule,
                            l1=self.l1, l2=self.l2, momentum=self.momentum)
        return self

    @staticmethod
    def onnx_shape_calculator():
        """
        Shape calculator when converting this model into ONNX.
        See :epkg:`skearn-onnx`.
        """
        from skl2onnx.common.data_types import Int64TensorType

        def shape_calculator(operator):
            op = operator.raw_operator
            input_type = operator.inputs[0].type.__class__
            input_dim = operator.inputs[0].get_first_dimension()
            output_type = input_type(
                [input_dim, op.estimator_.nodes[-1].ndim_out])
            operator.outputs[0].type = Int64TensorType([input_dim, 1])
            operator.outputs[1].type = output_type
        return shape_calculator

    @staticmethod
    def onnx_converter():
        """
        Converts this model into ONNX.
        """
        from skl2onnx.common.data_types import guess_numpy_type
        from skl2onnx.algebra.onnx_ops import (  # pylint: disable=E0611
            OnnxIdentity, OnnxArgMax, OnnxAdd, OnnxMatMul,
            OnnxSigmoid, OnnxMul, OnnxSoftmax)

        def converter(scope, operator, container):
            op = operator.raw_operator
            net = op.estimator_
            out = operator.outputs
            opv = container.target_opset

            X = operator.inputs[0]
            dtype = guess_numpy_type(X.type)

            res = {'inputs': X}
            last = None
            for node, attr in zip(net.nodes, net.nodes_attr):

                # verification
                coef = (node.coef.reshape((1, -1)) if len(node.coef.shape) == 1
                        else node.coef)
                if len(coef.shape) != 2:
                    raise RuntimeError(  # pragma: no cover
                        f"coef must be a 2D matrix not {coef.shape!r}.")
                if coef.shape[1] < 2:
                    raise RuntimeError(  # pragma: no cover
                        f"coef must be a 2D matrix with at least 2 columns "
                        f"not {coef.shape!r}.")

                # input, output, names
                name = ('inputs' if attr['inputs'][0] == 0 else 
                        "r_%s" % ("_".join(map(str, attr['inputs']))))
                if name not in res:
                    raise KeyError(  # pragma: no cover
                        f"Unable to find {name!r} in {set(res)}.")
                output_name = (
                    "r_%d" % attr['output'] if isinstance(attr['output'], int)
                    else "r_%s" % ("_".join(map(str, attr['output']))))
                x = res[name]

                # conversion of one node
                tr = OnnxAdd(OnnxMatMul(x, coef[:, 1:].T.astype(dtype),
                                        op_version=opv),
                             coef[:, 0].astype(dtype), op_version=opv)

                # activation
                if node.activation == "sigmoid4":
                    final = OnnxSigmoid(OnnxMul(tr, numpy.array([4], dtype=dtype),
                                                op_version=opv),
                                        op_version=opv)
                elif node.activation == "sigmoid":
                    final = OnnxSigmoid(tr, op_version=opv)
                elif node.activation == "softmax4":
                    final = OnnxSoftmax(OnnxMul(tr, numpy.array([4], dtype=dtype),
                                                op_version=opv),
                                        op_version=opv)
                elif node.activation == "softmax":
                    final = OnnxSoftmax(tr, op_version=opv)
                else:
                    raise NotImplementedError(
                        f"Unable to convert activation {node.activation!r} "
                        f"function into ONNX.")

                res[output_name] = final
                last = final

            prob = OnnxIdentity(last, op_version=opv, output_names=[out[1]])
            prob.add_to(scope, container)
            labels = OnnxArgMax(prob, axis=1, keepdims=1, op_version=opv,
                                output_names=[out[0]])
            labels.add_to(scope, container)
        return converter
