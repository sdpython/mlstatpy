# -*- coding: utf-8 -*-
"""
@brief      test log(time=6s)
"""
import io
import unittest
import pickle
import numpy
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


if __name__ == "__main__":
    unittest.main()
