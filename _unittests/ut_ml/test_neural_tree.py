import io
import unittest
import pickle
import numpy
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.datasets import load_iris
from sklearn.tree import export_text
from mlstatpy.ext_test_case import ExtTestCase, ignore_warnings
from onnx_array_api.plotting.text_plot import onnx_simple_text_plot
from mlstatpy.ml.neural_tree import (
    NeuralTreeNode,
    NeuralTreeNet,
    label_class_to_softmax_output,
    NeuralTreeNetClassifier,
    NeuralTreeNetRegressor,
)


class TestNeuralTree(ExtTestCase):
    def test_neural_tree_node(self):
        self.assertRaise(lambda: NeuralTreeNode([0, 1], 0.5, "identity2"))
        neu = NeuralTreeNode([0, 1], 0.5, "identity")
        res = neu.predict(numpy.array([4, 5]))
        self.assertEqual(res, 5.5)
        st = repr(neu)
        self.assertEqual(
            "NeuralTreeNode(weights=array([0., 1.]), "
            "bias=np.float64(0.5), activation='identity')",
            st,
        )
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
        self.assertEqualArray(exp.reshape((-1, 1)), got[:, -1:])
        rep = repr(net)
        self.assertEqual(rep, "NeuralTreeNet(3)")
        net.clear()
        self.assertEqual(len(net), 0)

    def test_neural_tree_network_append(self):
        net = NeuralTreeNet(3, empty=False)
        self.assertRaise(
            lambda: net.append(NeuralTreeNode(2, activation="identity"), inputs=[3])
        )
        net.append(NeuralTreeNode(1, activation="identity"), inputs=[3])
        self.assertEqual(net.size_, 5)
        last_node = net.nodes[-1]
        X = numpy.random.randn(2, 3)
        got = net.predict(X)
        exp = X.sum(axis=1) * last_node.input_weights[0] + last_node.bias
        self.assertEqual(exp.reshape((-1, 1)), got[:, -1:])
        rep = repr(net)
        self.assertEqual(rep, "NeuralTreeNet(3)")

    def test_neural_tree_network_copy(self):
        net = NeuralTreeNet(3, empty=False)
        net.append(NeuralTreeNode(1, activation="identity"), inputs=[3])
        net2 = net.copy()
        X = numpy.random.randn(2, 3)
        self.assertEqualArray(net.predict(X), net2.predict(X))

    def test_neural_tree_network_append_dim2(self):
        net = NeuralTreeNet(3, empty=False)
        self.assertRaise(
            lambda: net.append(NeuralTreeNode(2, activation="identity"), inputs=[3])
        )
        net.append(
            NeuralTreeNode(
                numpy.ones((2, 1), dtype=numpy.float64), activation="identity"
            ),
            inputs=[3],
        )
        self.assertEqual(net.size_, 6)
        last_node = net.nodes[-1]
        X = numpy.random.randn(2, 3)
        got = net.predict(X)
        exp = X.sum(axis=1) * last_node.input_weights[1, :] + last_node.bias[0]
        self.assertEqual(exp.reshape((-1,)), got[:, -2:-1].reshape((-1,)))
        exp = X.sum(axis=1) * last_node.input_weights[1, :] + last_node.bias[1]
        self.assertEqual(exp.reshape((-1,)), got[:, -1:].reshape((-1,)))
        rep = repr(net)
        self.assertEqual(rep, "NeuralTreeNet(3)")

    def test_convert(self):
        X = numpy.arange(8).astype(numpy.float64).reshape((-1, 2))
        y = ((X[:, 0] + X[:, 1] * 2) > 10).astype(numpy.int64)
        y2 = y.copy()
        y2[0] = 2

        tree = DecisionTreeClassifier(max_depth=2)
        tree.fit(X, y2)
        self.assertRaise(lambda: NeuralTreeNet.create_from_tree(tree), RuntimeError)

        tree = DecisionTreeClassifier(max_depth=2)
        tree.fit(X, y)
        root = NeuralTreeNet.create_from_tree(tree, 10)
        self.assertNotEmpty(root)
        exp = tree.predict_proba(X)
        got = root.predict(X)
        self.assertEqual(exp.shape[0], got.shape[0])
        self.assertEqualArray(exp, got[:, -2:], atol=1e-10)

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
        x[0, 3] = 1.0
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
        net.append(NeuralTreeNode(1, activation="identity"), inputs=[3])
        w = net.training_weights
        self.assertEqual(w.shape, (6,))
        self.assertEqual(w[0], 0)
        self.assertEqualArray(w[1:4], numpy.array([1, 1, 1], dtype=float))
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
        self.assertEqual(w.shape, (11,))
        delta = numpy.arange(11) + 0.5
        root.update_training_weights(delta)
        v2 = root.predict(X[:1])
        self.assertNotEqualArray(v1, v2)

    def test_gradients(self):
        X = numpy.array([0.1, 0.2, -0.3])
        w = numpy.array([10, 20, 3])
        g = numpy.array([-0.7], dtype=numpy.float64)
        for act in ["sigmoid", "sigmoid4", "expit", "identity", "relu", "leakyrelu"]:
            with self.subTest(act=act):
                neu = NeuralTreeNode(w, bias=-4, activation=act)
                pred = neu.predict(X)
                self.assertEqual(pred.shape, tuple())
                grad = neu.gradient_backward(g, X)
                self.assertEqual(grad.shape, (4,))
                grad = neu.gradient_backward(g, X, inputs=True)
                self.assertEqual(grad.shape, (3,))
                ww = neu.training_weights
                neu.update_training_weights(-ww)
                w0 = neu.training_weights
                self.assertEqualArray(w0, numpy.zeros(w0.shape))

        X = numpy.array([0.1, 0.2, -0.3])
        w = numpy.array([[10, 20, 3], [-10, -20, 3]])
        b = numpy.array([-3, 4], dtype=numpy.float64)
        g = numpy.array([-0.7, 0.2], dtype=numpy.float64)
        for act in ["softmax", "softmax4"]:
            with self.subTest(act=act):
                neu = NeuralTreeNode(w, bias=b, activation=act)
                pred = neu.predict(X)
                self.assertAlmostEqual(numpy.sum(pred), 1.0, atol=1e-10)
                self.assertEqual(pred.shape, (2,))
                grad = neu.gradient_backward(g, X)
                self.assertEqual(grad.shape, (2, 4))
                grad = neu.gradient_backward(g, X, inputs=True)
                self.assertEqual(grad.shape, (3,))
                ww = neu.training_weights
                neu.update_training_weights(-ww)
                w0 = neu.training_weights
                self.assertEqualArray(w0, numpy.zeros(w0.shape))

    def test_optim_regression(self):
        state = numpy.random.RandomState(seed=0)
        X = numpy.abs(state.randn(10, 2))
        w0 = state.randn(3)
        w1 = numpy.array([-0.5, 0.8, -0.6])
        noise = state.randn(X.shape[0]) / 10
        noise[0] = 0
        noise[1] = 0.07
        X[1, 0] = 0.7
        X[1, 1] = -0.5
        y = w1[0] + X[:, 0] * w1[1] + X[:, 1] * w1[2] + noise

        for act in ["identity", "relu", "leakyrelu", "sigmoid", "sigmoid4", "expit"]:
            with self.subTest(act=act):
                neu = NeuralTreeNode(w1[1:], bias=w1[0], activation=act)
                loss = neu.loss(X, y).sum() / X.shape[0]
                if act == "identity":
                    self.assertGreater(loss, 0)
                    self.assertLess(loss, 0.1)
                grad = neu.gradient(X[0], y[0])
                if act == "identity":
                    self.assertEqualArray(grad, numpy.zeros(grad.shape))
                grad = neu.gradient(X[1], y[1])
                ming, maxg = grad[:2].min(), grad[:2].max()
                if ming == maxg:
                    raise AssertionError(f"Gradient is wrong\nloss={loss}\ngrad={grad}")
                self.assertEqual(grad.shape, w0.shape)

                neu.fit(X, y, verbose=False)
                c1 = neu.training_weights
                neu = NeuralTreeNode(w0[1:], bias=w0[0], activation=act)
                neu.fit(X, y, verbose=False, lr_schedule="constant")
                c2 = neu.training_weights
                diff = numpy.abs(c2 - c1)
                if act == "identity":
                    self.assertLess(diff.max(), 0.16)

    def test_optim_clas(self):
        X = numpy.abs(numpy.random.randn(10, 2))
        w1 = numpy.array([[0.1, 0.8, -0.6], [-0.1, 0.4, -0.3]])
        w0 = numpy.random.randn(*w1.shape)
        noise = numpy.random.randn(*X.shape) / 10
        noise[0] = 0
        noise[1] = 0.07
        y0 = X[:, :1] @ w1[:, 1:2].T + X[:, 1:] @ w1[:, 2:3].T + w1[:, 0].T + noise
        y = numpy.exp(y0)
        y /= numpy.sum(y, axis=1, keepdims=1)
        y[:-1, 0] = (y[:-1, 0] >= 0.5).astype(numpy.float64)
        y[:-1, 1] = (y[:-1, 1] >= 0.5).astype(numpy.float64)
        y /= numpy.sum(y, axis=1, keepdims=1)

        for act in ["softmax", "softmax4"]:
            with self.subTest(act=act):
                neu2 = NeuralTreeNode(2, activation=act)
                neu = NeuralTreeNode(w1[:, 1:], bias=w1[:, 0], activation=act)
                self.assertEqual(
                    neu2.training_weights.shape, neu.training_weights.shape
                )
                self.assertEqual(neu2.input_weights.shape, neu.input_weights.shape)
                loss = neu.loss(X, y).sum() / X.shape[0]
                self.assertNotEmpty(loss)
                self.assertFalse(numpy.isinf(loss))
                self.assertFalse(numpy.isnan(loss))
                grad = neu.gradient(X[0], y[0])
                self.assertEqual(grad.ravel().shape, w1.ravel().shape)

                neu.fit(X, y, verbose=False)
                c1 = neu.training_weights
                neu = NeuralTreeNode(w0[:, 1:], bias=w0[:, 0], activation=act)
                neu.fit(X, y, verbose=False, lr_schedule="constant")
                c2 = neu.training_weights
                self.assertEqual(c1.shape, c2.shape)

    def test_label_class_to_softmax_output(self):
        y_label = numpy.array([0, 1, 0, 0])
        self.assertRaise(
            lambda: label_class_to_softmax_output(y_label.reshape((-1, 1))), ValueError
        )
        soft_y = label_class_to_softmax_output(y_label)
        self.assertEqual(soft_y.shape, (4, 2))
        self.assertEqualArray(soft_y[:, 1], y_label.astype(float))
        self.assertEqualArray(soft_y[:, 0], 1 - y_label.astype(float))

    def test_neural_net_gradient(self):
        X = numpy.arange(8).astype(numpy.float64).reshape((-1, 2))
        y = ((X[:, 0] + X[:, 1] * 2) > 10).astype(numpy.int64)
        ny = label_class_to_softmax_output(y)

        tree = DecisionTreeClassifier(max_depth=2)
        tree.fit(X, y)
        root = NeuralTreeNet.create_from_tree(tree, 10)
        _, out, err = self.capture(lambda: root.fit(X, ny, verbose=True))
        self.assertIn("loss:", out)
        self.assertEmpty(err)

    def test_neural_net_gradient_regression(self):
        X = numpy.abs(numpy.random.randn(10, 2))
        w1 = numpy.array([-0.5, 0.8, -0.6])
        noise = numpy.random.randn(X.shape[0]) / 10
        noise[0] = 0
        noise[1] = 0.07
        X[1, 0] = 0.7
        X[1, 1] = -0.5
        y = w1[0] + X[:, 0] * w1[1] + X[:, 1] * w1[2] + noise

        for act in ["identity", "relu", "leakyrelu", "sigmoid", "sigmoid4", "expit"]:
            with self.subTest(act=act):
                neu = NeuralTreeNode(w1[1:], bias=w1[0], activation=act)
                loss1 = neu.loss(X, y)
                grad1 = neu.gradient(X[0], y[0])

                net = NeuralTreeNet(X.shape[1], empty=True)
                net.append(neu, numpy.arange(0, 2))
                loss2 = net.loss(X, y)
                grad2 = net.gradient(X[0], y[0])
                self.assertEqualArray(loss1, loss2, atol=1e-5)
                self.assertEqualArray(grad1, grad2, atol=1e-5)

    @ignore_warnings(DeprecationWarning)
    def test_neural_net_gradient_regression_2(self):
        X = numpy.abs(numpy.random.randn(10, 2))
        w1 = numpy.array([-0.5, 0.8, -0.6])
        noise = numpy.random.randn(X.shape[0]) / 10
        noise[0] = 0
        noise[1] = 0.07
        X[1, 0] = 0.7
        X[1, 1] = -0.5
        y = w1[0] + X[:, 0] * w1[1] + X[:, 1] * w1[2] + noise

        for act in ["relu", "sigmoid", "identity", "leakyrelu", "sigmoid4", "expit"]:
            with self.subTest(act=act):
                neu = NeuralTreeNode(w1[1:], bias=w1[0], activation=act)
                loss1 = neu.loss(X, y)
                pred1 = neu.predict(X)
                if act == "relu":
                    self.assertEqualArray(pred1[1:2], numpy.array([0.36]))
                    pred11 = neu.predict(X)
                    self.assertEqualArray(pred11[1:2], numpy.array([0.36]))

                net = NeuralTreeNet(X.shape[1], empty=True)
                net.append(neu, numpy.arange(0, 2))
                ide = NeuralTreeNode(
                    numpy.array([1], dtype=X.dtype),
                    bias=numpy.array([0], dtype=X.dtype),
                    activation="identity",
                )
                net.append(ide, numpy.arange(2, 3))
                pred2 = net.predict(X)
                loss2 = net.loss(X, y)

                self.assertEqualArray(pred1, pred2[:, -1], atol=1e-10)
                self.assertEqualArray(pred2[:, -2], pred2[:, -1])
                self.assertEqualArray(pred2[:, 2], pred2[:, 3])
                self.assertEqualArray(loss1, loss2, atol=1e-7)

                for p in range(5):
                    grad1 = neu.gradient(X[p], y[p])
                    grad2 = net.gradient(X[p], y[p])
                    self.assertEqualArray(grad1, grad2[:3], atol=1e-7)

    @ignore_warnings(DeprecationWarning)
    def test_neural_net_gradient_regression_2_h2(self):
        X = numpy.abs(numpy.random.randn(10, 2))
        w1 = numpy.array([-0.5, 0.8, -0.6])
        noise = numpy.random.randn(X.shape[0]) / 10
        noise[0] = 0
        noise[1] = 0.07
        X[1, 0] = 0.7
        X[1, 1] = -0.5
        y = w1[0] + X[:, 0] * w1[1] + X[:, 1] * w1[2] + noise

        for act in ["relu", "sigmoid", "identity", "leakyrelu", "sigmoid4", "expit"]:
            with self.subTest(act=act):
                neu = NeuralTreeNode(w1[1:], bias=w1[0], activation=act)
                loss1 = neu.loss(X, y)
                pred1 = neu.predict(X)
                if act == "relu":
                    self.assertEqualArray(pred1[1:2], numpy.array([0.36]))
                    pred11 = neu.predict(X)
                    self.assertEqualArray(pred11[1:2], numpy.array([0.36]))

                net = NeuralTreeNet(X.shape[1], empty=True)
                net.append(neu, numpy.arange(0, 2))

                # a layer of identity neurons

                ide1 = NeuralTreeNode(
                    numpy.array([0.7], dtype=X.dtype),
                    bias=numpy.array([0], dtype=X.dtype),
                    activation="identity",
                )
                net.append(ide1, numpy.arange(2, 3))

                ide2 = NeuralTreeNode(
                    numpy.array([0.3], dtype=X.dtype),
                    bias=numpy.array([0], dtype=X.dtype),
                    activation="identity",
                )
                net.append(ide2, numpy.arange(2, 3))

                # sums of the two last neurons

                ide3 = NeuralTreeNode(
                    numpy.array([1, 1], dtype=X.dtype),
                    bias=numpy.array([0], dtype=X.dtype),
                    activation="identity",
                )
                net.append(ide3, numpy.arange(3, 5))

                # same verification
                pred2 = net.predict(X)
                loss2 = net.loss(X, y)

                self.assertEqualArray(pred1, pred2[:, -1], atol=1e-8)
                self.assertEqualArray(pred2[:, 2], pred2[:, -1], atol=1e-10)
                self.assertEqualArray(loss1, loss2, atol=1e-7)

                for p in range(5):
                    grad1 = neu.gradient(X[p], y[p])
                    grad2 = net.gradient(X[p], y[p])
                    self.assertEqualArray(grad1, grad2[:3], atol=1e-7)

                loss1 = net.loss(X, y).sum()
                net.fit(X, y, max_iter=20)
                loss2 = net.loss(X, y).sum()
                self.assertLess(loss2, loss1 + 1e-7)

    def test_neural_net_gradient_fit(self):
        X = numpy.arange(16).astype(numpy.float64).reshape((-1, 2))
        y = ((X[:, 0] + X[:, 1] * 2) > 15).astype(numpy.int64)
        ny = label_class_to_softmax_output(y)

        tree = DecisionTreeClassifier(max_depth=2)
        tree.fit(X, y)
        root = NeuralTreeNet.create_from_tree(tree, 10)
        loss1 = root.loss(X, ny).sum()
        self.assertGreater(loss1, -1e-5)
        self.assertLess(loss1, 1.0)
        _, out, err = self.capture(lambda: root.fit(X, ny, verbose=True, max_iter=20))
        self.assertEmpty(err)
        self.assertNotEmpty(out)
        loss2 = root.loss(X, ny).sum()
        self.assertLess(loss2, loss1 + 1)

    def test_neural_net_gradient_fit2(self):
        X = numpy.arange(16).astype(numpy.float64).reshape((-1, 2))
        y = ((X[:, 0] + X[:, 1] * 2) > 15).astype(numpy.int64)
        y[0] = 1
        y[-1] = 0
        ny = label_class_to_softmax_output(y)

        tree = DecisionTreeClassifier(max_depth=2)
        tree.fit(X, y)
        self.assertLess(tree.score(X, y), 1)
        root = NeuralTreeNet.create_from_tree(tree, 0.01)
        loss1 = root.loss(X, ny).sum()
        self.assertGreater(loss1, -1e-5)
        self.assertLess(loss1, 60.0)
        _, out, err = self.capture(
            lambda: root.fit(X, ny, verbose=True, max_iter=20, l2=0.001, momentum=0.1)
        )
        self.assertNotEmpty(out)
        self.assertEmpty(err)
        loss2 = root.loss(X, ny).sum()
        self.assertLess(loss2, loss1 + 100)

    def test_shape_dim2(self):
        X = numpy.random.randn(10, 3)
        w = numpy.array([[10, 20, 3], [-10, -20, 0.5]])
        first = None
        for act in ["sigmoid", "sigmoid4", "expit", "identity", "relu", "leakyrelu"]:
            with self.subTest(act=act):
                neu = NeuralTreeNode(w, bias=[-4, 0.5], activation=act)
                pred = neu.predict(X)
                self.assertEqual(pred.shape, (X.shape[0], 2))
                text = str(neu)
                self.assertIn("NeuralTreeNode(", text)
                if first is None:
                    first = neu
                else:
                    self.assertFalse(neu == first)
                self.assertEqual(neu.ndim, 3)
                loss = neu.loss(X[0], 0.0)
                self.assertEqual(loss.shape, (2,))
                loss = neu.loss(X, numpy.zeros((X.shape[0], 1), dtype=numpy.float64))
                self.assertEqual(loss.shape, (10, 2))

    @ignore_warnings(DeprecationWarning)
    def test_convert_compact(self):
        X = numpy.arange(8).astype(numpy.float64).reshape((-1, 2))
        y = ((X[:, 0] + X[:, 1] * 2) > 10).astype(numpy.int64)
        y2 = y.copy()
        y2[0] = 2

        tree = DecisionTreeClassifier(max_depth=2)
        tree.fit(X, y2)
        self.assertRaise(
            lambda: NeuralTreeNet.create_from_tree(tree, arch="k"), ValueError
        )
        self.assertRaise(
            lambda: NeuralTreeNet.create_from_tree(tree, arch="compact"), RuntimeError
        )

        tree = DecisionTreeClassifier(max_depth=2)
        tree.fit(X, y)
        root = NeuralTreeNet.create_from_tree(tree, 10, arch="compact")
        self.assertNotEmpty(root)
        exp = tree.predict_proba(X)
        got = root.predict(X)
        self.assertEqual(exp.shape[0], got.shape[0])
        self.assertEqualArray(exp + 1e-8, got[:, -2:] + 1e-8)
        dot = root.to_dot()
        self.assertIn("s3a4:f4 -> s5a6:f6", dot)

    def test_convert_compact_fit(self):
        X = numpy.arange(8).astype(numpy.float64).reshape((-1, 2))
        y = ((X[:, 0] + X[:, 1] * 2) > 10).astype(numpy.int64)
        y2 = y.copy()
        y2[0] = 2

        tree = DecisionTreeClassifier(max_depth=2)
        tree.fit(X, y)
        root = NeuralTreeNet.create_from_tree(tree, 10, arch="compact")
        self.assertNotEmpty(root)
        exp = tree.predict_proba(X)
        got = root.predict(X)
        self.assertEqual(exp.shape[0], got.shape[0])
        self.assertEqualArray(exp + 1e-8, got[:, -2:] + 1e-8)
        ny = label_class_to_softmax_output(y)
        loss1 = root.loss(X, ny).sum()
        _, out, err = self.capture(lambda: root.fit(X, ny, verbose=True, max_iter=20))
        self.assertEmpty(err)
        self.assertNotEmpty(out)
        loss2 = root.loss(X, ny).sum()
        self.assertLess(loss2, loss1 + 1)

    def test_convert_compact_skl(self):
        X = numpy.arange(8).astype(numpy.float64).reshape((-1, 2))
        y = ((X[:, 0] + X[:, 1] * 2) > 10).astype(numpy.int64)
        tree = DecisionTreeClassifier(max_depth=2)
        tree.fit(X, y)
        root = NeuralTreeNet.create_from_tree(tree, 10, arch="compact")

        exp = tree.predict_proba(X)
        got = root.predict(X)
        self.assertEqual(exp.shape[0], got.shape[0])
        self.assertEqualArray(exp + 1e-8, got[:, -2:] + 1e-8)

        skl = NeuralTreeNetClassifier(root)
        prob = skl.predict_proba(X)
        self.assertEqualArray(exp, prob, atol=1e-10)
        lab = skl.predict(X)
        self.assertEqual(lab.shape, (X.shape[0],))

    def test_convert_compact_skl_fit(self):
        X = numpy.arange(8).astype(numpy.float64).reshape((-1, 2))
        y = ((X[:, 0] + X[:, 1] * 2) > 10).astype(numpy.int64)
        tree = DecisionTreeClassifier(max_depth=2)
        tree.fit(X, y)
        root = NeuralTreeNet.create_from_tree(tree, 10, arch="compact")
        skl = NeuralTreeNetClassifier(root)
        skl.fit(X, y)
        exp = tree.predict_proba(X)
        got = skl.predict_proba(X)
        self.assertEqualArray(exp, got, atol=1e-10)

    def test_convert_compact_skl_onnx(self):
        from skl2onnx import to_onnx
        from onnx.reference import ReferenceEvaluator

        X = numpy.arange(8).astype(numpy.float64).reshape((-1, 2))
        y = ((X[:, 0] + X[:, 1] * 2) > 10).astype(numpy.int64)
        tree = DecisionTreeClassifier(max_depth=3)
        tree.fit(X, y)
        root = NeuralTreeNet.create_from_tree(tree, 10, arch="compact")
        skl = NeuralTreeNetClassifier(root)
        got = skl.predict_proba(X)
        exp = tree.predict_proba(X)
        self.assertEqualArray(exp, got, atol=1e-10)
        dec = root.predict(X)
        self.assertEqualArray(exp, dec[:, -2:], atol=1e-10)

        x32 = X.astype(numpy.float32)
        onx = to_onnx(skl, x32, target_opset=15)
        text = onnx_simple_text_plot(onx)
        self.assertIn("Sigmoid(", text)
        self.assertIn("Softmax(", text)
        oinf = ReferenceEvaluator(onx)
        got2 = oinf.run(None, {"X": x32})[0]
        self.assertEqualArray(exp[:, 1], got2.astype(float).ravel(), atol=1e-5)

    @ignore_warnings(DeprecationWarning)
    def test_convert_reg_compact(self):
        X = numpy.arange(32).astype(numpy.float64).reshape((-1, 2))
        y = (X[:, 0] + X[:, 1] * 2).astype(numpy.float64)
        tree = DecisionTreeRegressor(max_depth=3)
        tree.fit(X, y)
        text = export_text(tree, feature_names=["x1", "x2"])
        self.assertIn("[5.00]", text)
        root = NeuralTreeNet.create_from_tree(tree, 10, arch="compact")
        # if __name__ == '__main__':
        #     print(text)
        #     for n in root.nodes:
        #         print(n)
        #     print('--------------')
        #     t = X[2:3]
        #     print(t)
        #     for n in root.nodes:
        #         print('*')
        #         ii = n._predict(t)
        #         print((ii * 10 + 0.01).astype(numpy.int64) / 10.)
        #         h = n.predict(t)
        #         print((h * 10 + 0.01).astype(numpy.int64) / 10.)
        #         t = h
        self.assertNotEmpty(root)
        exp = tree.predict(X)
        got = root.predict(X)
        self.assertEqualArray(exp, got[:, -1], atol=1e-6)
        dot = root.to_dot()
        self.assertIn("9 -> 17", dot)

    @ignore_warnings(DeprecationWarning)
    def test_convert_compact_skl_reg(self):
        X = numpy.arange(8).astype(numpy.float64).reshape((-1, 2))
        y = X[:, 0] + X[:, 1] * 2
        tree = DecisionTreeRegressor(max_depth=2)
        tree.fit(X, y)
        root = NeuralTreeNet.create_from_tree(tree, 10, arch="compact")

        exp = tree.predict(X)
        got = root.predict(X)
        self.assertEqual(exp.shape[0], got.shape[0])
        self.assertEqualArray(exp, got[:, -1], atol=1e-7)

        skl = NeuralTreeNetRegressor(root)
        prob = skl.predict(X)
        self.assertEqualArray(exp, prob.ravel(), atol=1e-7)

    @ignore_warnings(DeprecationWarning)
    def test_convert_compact_skl_fit_reg(self):
        X = numpy.arange(8).astype(numpy.float64).reshape((-1, 2))
        y = X[:, 0] + X[:, 1] * 2
        tree = DecisionTreeRegressor(max_depth=2)
        tree.fit(X, y)
        root = NeuralTreeNet.create_from_tree(tree, 10, arch="compact")
        skl = NeuralTreeNetRegressor(root)
        skl.fit(X, y)
        exp = tree.predict(X)
        got = skl.predict(X)
        self.assertEqualArray(exp, got.ravel(), atol=1e-7)

    @ignore_warnings(DeprecationWarning)
    def test_convert_compact_skl_onnx_reg(self):
        from skl2onnx import to_onnx
        from onnx.reference import ReferenceEvaluator

        X = numpy.arange(8).astype(numpy.float64).reshape((-1, 2))
        y = X[:, 0] + X[:, 1] * 2
        tree = DecisionTreeRegressor(max_depth=3)
        tree.fit(X, y)
        root = NeuralTreeNet.create_from_tree(tree, 10, arch="compact")
        skl = NeuralTreeNetRegressor(root)
        got = skl.predict(X)
        exp = tree.predict(X)
        self.assertEqualArray(exp, got.ravel(), atol=1e-7)
        dec = root.predict(X)
        self.assertEqualArray(exp, dec[:, -1], atol=1e-7)

        x32 = X.astype(numpy.float32)
        onx = to_onnx(skl, x32, target_opset=15)
        text = onnx_simple_text_plot(onx)
        self.assertIn("Sigmoid(", text)
        self.assertNotIn("Softmax(", text)
        oinf = ReferenceEvaluator(onx)
        got2 = oinf.run(None, {"X": x32})[0]
        self.assertEqualArray(exp, got2.ravel().astype(float))


if __name__ == "__main__":
    unittest.main(verbosity=2)
