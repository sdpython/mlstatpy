# -*- coding: utf-8 -*-
"""
@brief      test log(time=6s)
"""
import io
import unittest
import pickle
import numpy
from sklearn.tree import DecisionTreeClassifier
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
        net = NeuralTreeNet(3)
        X = numpy.random.randn(2, 3)
        got = net.predict(X)
        exp = X.sum(axis=1)
        self.assertEqual(exp.reshape((-1, 1)), got[:, -1:])
        rep = repr(net)
        self.assertEqual(rep, 'NeuralTreeNet(3)')

    def test_neural_tree_network_append(self):
        net = NeuralTreeNet(3)
        self.assertRaise(
            lambda: net.append(
                NeuralTreeNode(2, activation='identity'), inputs=[3]))
        net.append(NeuralTreeNode(1, activation='identity'),
                   inputs=[3])
        self.assertEqual(net.size_, 5)
        last_node = net.nodes[-1]
        X = numpy.random.randn(2, 3)
        got = net.predict(X)
        exp = X.sum(axis=1) * last_node.weights[0] + last_node.bias
        self.assertEqual(exp.reshape((-1, 1)), got[:, -1:])
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
        root = NeuralTreeNet.create_from_tree(tree)
        self.assertNotEmpty(root)
        exp = tree.predict(X)
        got = root.predict(X)
        self.assertEqual(exp.shape[0], got.shape[0])


if __name__ == "__main__":
    unittest.main()
