# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import io
import unittest
import pickle
import numpy
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.datasets import load_iris
from pyquickhelper.pycode import ExtTestCase
from mlstatpy.ml.neural_tree import NeuralTreeNode, NeuralTreeNet


class TestNeuralTree(ExtTestCase):

    def test_neural_tree_node(self):
        self.assertRaise(lambda: NeuralTreeNode([0, 1], 0.5, 'identity2'))
        neu = NeuralTreeNode([0, 1], 0.5, 'identity')
        res = neu.predict(numpy.array([4, 5]))
        self.assertEqual(res, 5.5)
        st = repr(neu)
        self.assertEqual("NeuralTreeNode(weights=array([0., 1.]), "
                         "bias=0.5, activation='identity')", st)
        st = io.BytesIO()
        pickle.dump(neu, st)
        st = io.BytesIO(st.getvalue())
        neu2 = pickle.load(st)
        self.assertTrue(neu == neu2)

    def test_neural_tree_network(self):
        net = NeuralTreeNet(3, empty=False)
        X = numpy.random.randn(2, 3)
        got = net.predict(X)
        exp = X.sum(axis=1)
        self.assertEqual(exp.reshape((-1, 1)), got[:, -1:])
        rep = repr(net)
        self.assertEqual(rep, 'NeuralTreeNet(3)')
        net.clear()
        self.assertEqual(len(net), 0)

    def test_neural_tree_network_append(self):
        net = NeuralTreeNet(3, empty=False)
        self.assertRaise(
            lambda: net.append(
                NeuralTreeNode(2, activation='identity'), inputs=[3]))
        net.append(NeuralTreeNode(1, activation='identity'),
                   inputs=[3])
        self.assertEqual(net.size_, 5)
        last_node = net.nodes[-1]
        X = numpy.random.randn(2, 3)
        got = net.predict(X)
        exp = X.sum(axis=1) * last_node.input_weights[0] + last_node.bias
        self.assertEqual(exp.reshape((-1, 1)), got[:, -1:])
        rep = repr(net)
        self.assertEqual(rep, 'NeuralTreeNet(3)')

    def test_neural_tree_network_append_dim2(self):
        net = NeuralTreeNet(3, empty=False)
        self.assertRaise(
            lambda: net.append(
                NeuralTreeNode(2, activation='identity'), inputs=[3]))
        net.append(NeuralTreeNode(numpy.ones((2, 1), dtype=numpy.float64),
                                  activation='identity'),
                   inputs=[3])
        self.assertEqual(net.size_, 6)
        last_node = net.nodes[-1]
        X = numpy.random.randn(2, 3)
        got = net.predict(X)
        exp = X.sum(axis=1) * last_node.input_weights[1, :] + last_node.bias[0]
        self.assertEqual(exp.reshape((-1, )), got[:, -2: -1].reshape((-1, )))
        exp = X.sum(axis=1) * last_node.input_weights[1, :] + last_node.bias[1]
        self.assertEqual(exp.reshape((-1, )), got[:, -1:].reshape((-1, )))
        rep = repr(net)
        self.assertEqual(rep, 'NeuralTreeNet(3)')

    def test_convert(self):
        X = numpy.arange(8).astype(numpy.float64).reshape((-1, 2))
        y = ((X[:, 0] + X[:, 1] * 2) > 10).astype(numpy.int64)
        y2 = y.copy()
        y2[0] = 2

        tree = DecisionTreeClassifier(max_depth=2)
        tree.fit(X, y2)
        self.assertRaise(
            lambda: NeuralTreeNet.create_from_tree(tree), RuntimeError)

        tree = DecisionTreeClassifier(max_depth=2)
        tree.fit(X, y)
        root = NeuralTreeNet.create_from_tree(tree, 10)
        self.assertNotEmpty(root)
        exp = tree.predict_proba(X)
        got = root.predict(X)
        self.assertEqual(exp.shape[0], got.shape[0])
        self.assertEqualArray(exp, got[:, -2:])

    def test_dot(self):
        data = load_iris()
        X, y = data.data, data.target
        y = y % 2

        tree = DecisionTreeClassifier(max_depth=3, random_state=11)
        tree.fit(X, y)
        root = NeuralTreeNet.create_from_tree(tree)
        dot = export_graphviz(tree)
        self.assertIn("digraph", dot)

        dot2 = root.to_dot()
        self.assertIn("digraph", dot2)
        x = X[:1].copy()
        x[0, 3] = 1.
        dot2 = root.to_dot(X=x.ravel())
        self.assertIn("digraph", dot2)
        exp = tree.predict_proba(X)[:, -1]
        got = root.predict(X)[:, -1]
        mat = numpy.empty((exp.shape[0], 2), dtype=exp.dtype)
        mat[:, 0] = exp
        mat[:, 1] = got
        c = numpy.corrcoef(mat.T)
        self.assertGreater(c[0, 1], 0.5)

    def test_neural_tree_network_training_weights(self):
        net = NeuralTreeNet(3, empty=False)
        net.append(NeuralTreeNode(1, activation='identity'),
                   inputs=[3])
        w = net.training_weights
        self.assertEqual(w.shape, (6, ))
        self.assertEqual(w[0], 0)
        self.assertEqualArray(w[1:4], [1, 1, 1])
        delta = numpy.arange(6) - 0.5
        net.update_training_weights(delta)
        w2 = net.training_weights
        self.assertEqualArray(w2, w + delta)

    def test_training_weights(self):
        X = numpy.arange(8).astype(numpy.float64).reshape((-1, 2))
        y = ((X[:, 0] + X[:, 1] * 2) > 10).astype(numpy.int64)
        y2 = y.copy()
        y2[0] = 2

        tree = DecisionTreeClassifier(max_depth=2)
        tree.fit(X, y)
        root = NeuralTreeNet.create_from_tree(tree, 10)
        v1 = root.predict(X[:1])
        w = root.training_weights
        self.assertEqual(w.shape, (11, ))
        delta = numpy.arange(11) + 0.5
        root.update_training_weights(delta)
        v2 = root.predict(X[:1])
        self.assertNotEqualArray(v1, v2)

    def test_gradients(self):
        X = numpy.array([0.1, 0.2, -0.3])
        w = numpy.array([10, 20, 3])
        g = numpy.array([-0.7], dtype=numpy.float64)
        z = numpy.zeros((4, ), dtype=w.dtype)
        for act in ['sigmoid', 'sigmoid4', 'expit', 'identity', 'relu']:
            neu = NeuralTreeNode(w, bias=-4, activation=act)
            pred = neu.predict(X)
            self.assertEqual(pred.shape, tuple())
            grad = neu.gradient_backward(g, z, X)
            self.assertEqual(grad.shape, (4, ))
            grad = neu.gradient_backward(g, z, X, inputs=True)
            self.assertEqual(grad.shape, (3, ))
            ww = neu.training_weights
            neu.update_training_weights(-ww)
            w0 = neu.training_weights
            self.assertEqualArray(w0, numpy.zeros(w0.shape))

        X = numpy.array([0.1, 0.2, -0.3])
        w = numpy.array([[10, 20, 3], [-10, -20, 3]])
        b = numpy.array([-3, 4], dtype=numpy.float64)
        g = numpy.array([-0.7, 0.2], dtype=numpy.float64)
        z = numpy.zeros((2, 4), dtype=w.dtype)
        for act in ['softmax', 'softmax4']:
            neu = NeuralTreeNode(w, bias=b, activation=act)
            pred = neu.predict(X)
            self.assertAlmostEqual(numpy.sum(pred), 1.)
            self.assertEqual(pred.shape, (2,))
            grad = neu.gradient_backward(g, z, X)
            self.assertEqual(grad.shape, (2, 4))
            grad = neu.gradient_backward(g, z, X, inputs=True)
            self.assertEqual(grad.shape, (3, ))
            ww = neu.training_weights
            neu.update_training_weights(-ww)
            w0 = neu.training_weights
            self.assertEqualArray(w0, numpy.zeros(w0.shape))

    def test_optim_regression(self):
        X = numpy.random.randn(10, 2)
        y = numpy.sum(X, axis=1) + numpy.random.randn(X.shape[0]) / 10
        w0 = numpy.random.randn(3)
        for act in ['sigmoid', 'sigmoid4', 'expit', 'identity', 'relu']:
            neu = NeuralTreeNode(w0[1:], bias=w0[0], activation=act)

            xt, yt = X[0], y[0]
            try:
                grad = neu.gradient(xt, yt)
            except ValueError as e:
                raise ValueError("Issue with act='{}'.".format(act)) from e
            self.assertEqual(grad.shape, w0.shape)


if __name__ == "__main__":
    TestNeuralTree().test_optim_regression()
    unittest.main()
